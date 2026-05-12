"""MA-DBP ROS 2 Action Server node.

Exposes a ``madbp_interfaces/action/PlanMultiAgentPath`` action server.
On a goal request it:
  1. Receives start/goal poses and n_agents from the goal message.
  2. Grabs the latest rendered env image from the ``/env/image`` topic
     (published by lichtblik_bridge or mujoco_bridge).
  3. Runs ``MultiAgentDiffusionPlanner.sample_moving_window`` in a
     background thread so the ROS executor stays responsive.
  4. Publishes per-agent nav_msgs/Path on ``/madbp/agent_{i}/plan``.
  5. Returns the full trajectory tensor as the action result.

Parameters (ROS 2 params, all overridable via a YAML param file):
  checkpoint     : str   — path to .pth checkpoint
  max_agents     : int   = 10
  horizon        : int   = 8
  diffusion_steps: int   = 150
  cfg_scale      : float = 1.0   (1.0 = no CFG)
  max_step_size  : float = 0.05
  device         : str   = "cpu"
  image_topic    : str   = "/env/image"
  plan_topic_base: str   = "/madbp/agent"
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import torch

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from madbp.models.diffusion import MultiAgentDiffusionPlanner

# Action interface — defined in the companion madbp_interfaces package.
# Import lazily so the file can be parsed without it installed.
try:
    from madbp_interfaces.action import PlanMultiAgentPath  # type: ignore
except ImportError:  # pragma: no cover
    PlanMultiAgentPath = None  # type: ignore


class MADBPPlannerNode(Node):
    """ROS 2 node wrapping the MA-DBP moving-window sampler."""

    def __init__(self) -> None:
        super().__init__("madbp_planner")

        # --- ROS params ---------------------------------------------------
        self.declare_parameter("checkpoint", "")
        self.declare_parameter("max_agents", 10)
        self.declare_parameter("horizon", 8)
        self.declare_parameter("diffusion_steps", 150)
        self.declare_parameter("cfg_scale", 1.0)
        self.declare_parameter("max_step_size", 0.05)
        self.declare_parameter("device", "cpu")
        self.declare_parameter("image_topic", "/env/image")
        self.declare_parameter("plan_topic_base", "/madbp/agent")
        self.declare_parameter("full_horizon_length", 64)

        ckpt_path: str = self.get_parameter("checkpoint").value
        self._device = torch.device(self.get_parameter("device").value)
        self._cfg_scale: float = self.get_parameter("cfg_scale").value
        self._max_step: float = self.get_parameter("max_step_size").value
        self._full_h: int = self.get_parameter("full_horizon_length").value
        plan_base: str = self.get_parameter("plan_topic_base").value
        max_agents: int = self.get_parameter("max_agents").value

        # --- Load model ---------------------------------------------------
        if not ckpt_path:
            self.get_logger().warn(
                "No checkpoint provided — model is randomly initialised. "
                "Set the 'checkpoint' parameter for real evaluation."
            )
            self._model = MultiAgentDiffusionPlanner(
                max_agents=max_agents,
                horizon=self.get_parameter("horizon").value,
                diffusion_steps=self.get_parameter("diffusion_steps").value,
            ).to(self._device)
        else:
            ckpt = torch.load(ckpt_path, map_location=self._device)
            cfg = ckpt.get("cfg", {})
            self._model = MultiAgentDiffusionPlanner(
                max_agents=cfg.get("max_agents", max_agents),
                horizon=cfg.get("horizon", self.get_parameter("horizon").value),
                diffusion_steps=cfg.get("diffusion_steps", self.get_parameter("diffusion_steps").value),
                cfg_dropout_prob=cfg.get("cfg_dropout_prob", 0.0),
            ).to(self._device)
            self._model.load_state_dict(ckpt["model"])
        self._model.eval()

        # --- Latest env image (written by bridge node) --------------------
        self._latest_image: torch.Tensor | None = None
        self._image_lock = threading.Lock()
        image_topic: str = self.get_parameter("image_topic").value
        self._img_sub = self.create_subscription(
            Image, image_topic, self._image_callback, 10
        )

        # --- Per-agent path publishers ------------------------------------
        self._path_pubs: dict[int, Any] = {}
        for i in range(max_agents):
            topic = f"{plan_base}_{i}/plan"
            self._path_pubs[i] = self.create_publisher(Path, topic, 10)

        # --- Action server -----------------------------------------------
        if PlanMultiAgentPath is not None:
            self._action_server = ActionServer(
                self,
                PlanMultiAgentPath,
                "plan_multi_agent_path",
                execute_callback=self._execute_callback,
                goal_callback=self._goal_callback,
                cancel_callback=self._cancel_callback,
            )
            self.get_logger().info("MA-DBP planner action server ready.")
        else:
            self.get_logger().warn(
                "madbp_interfaces not found — action server disabled. "
                "Build the madbp_interfaces package first."
            )

    # ------------------------------------------------------------------
    # Image subscription
    # ------------------------------------------------------------------

    def _image_callback(self, msg: Image) -> None:
        """Convert ROS Image → normalised float32 tensor [1,3,H,W]."""
        H, W = msg.height, msg.width
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(H, W, -1)
        t = torch.from_numpy(arr[:, :, :3].copy()).float() / 255.0
        t = t.permute(2, 0, 1).unsqueeze(0).to(self._device)
        with self._image_lock:
            self._latest_image = t

    # ------------------------------------------------------------------
    # Action server callbacks
    # ------------------------------------------------------------------

    def _goal_callback(self, _goal_request: Any) -> GoalResponse:
        return GoalResponse.ACCEPT

    def _cancel_callback(self, _goal_handle: Any) -> CancelResponse:
        return CancelResponse.ACCEPT

    def _execute_callback(self, goal_handle: Any) -> Any:
        """Run the MA-DBP planner and publish per-agent Path messages."""
        req = goal_handle.request
        n_agents: int = int(req.n_agents)
        max_agents = self._model.max_agents

        # Starts / goals from the action goal [n_agents, 2]
        starts_np = np.array([[p.x, p.y] for p in req.starts], dtype=np.float32)
        goals_np = np.array([[p.x, p.y] for p in req.goals], dtype=np.float32)

        starts_t = torch.zeros(1, max_agents, 2, device=self._device)
        goals_t = torch.zeros(1, max_agents, 2, device=self._device)
        starts_t[0, :n_agents] = torch.from_numpy(starts_np).to(self._device)
        goals_t[0, :n_agents] = torch.from_numpy(goals_np).to(self._device)
        n_agents_t = torch.tensor([n_agents], device=self._device)

        # Grab env image (fall back to blank if not yet received)
        with self._image_lock:
            frame = self._latest_image
        if frame is None:
            self.get_logger().warn("No env image received yet — using blank frame.")
            frame = torch.zeros(1, 3, 64, 64, device=self._device)

        # Plan (runs in executor thread — already async from action server)
        self.get_logger().info(f"Planning for {n_agents} agents over horizon {self._full_h}...")
        with torch.no_grad():
            traj = self._model.sample_moving_window(
                frame, starts_t, goals_t, n_agents_t,
                full_horizon_length=self._full_h,
                guidance_scale=self._cfg_scale,
                max_step_size=self._max_step,
            )  # [1, max_agents, full_h, 2]

        # Publish nav_msgs/Path for each active agent
        stamp = self.get_clock().now().to_msg()
        for i in range(n_agents):
            path_msg = Path()
            path_msg.header = Header(stamp=stamp, frame_id="map")
            for t_step in range(traj.shape[2]):
                ps = PoseStamped()
                ps.header = path_msg.header
                ps.pose.position.x = float(traj[0, i, t_step, 0])
                ps.pose.position.y = float(traj[0, i, t_step, 1])
                ps.pose.orientation.w = 1.0
                path_msg.poses.append(ps)
            self._path_pubs[i].publish(path_msg)

        goal_handle.succeed()
        result = PlanMultiAgentPath.Result()
        result.success = True
        self.get_logger().info("Planning complete.")
        return result


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MADBPPlannerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
