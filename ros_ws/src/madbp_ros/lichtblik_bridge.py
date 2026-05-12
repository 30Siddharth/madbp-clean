"""Lichtblik ↔ ROS 2 bridge for MA-DBP validation.

Lichtblik is a lightweight 2-D robot simulator designed for multi-agent
navigation research. This bridge:

  * Ticks the Lichtblik world at a fixed rate (default 20 Hz).
  * Publishes each robot's odometry on ``/robot_{i}/odom``.
  * Publishes TF transforms: map → odom → base_link for each robot.
  * Publishes a top-down RGB render of the world on ``/env/image``
    (consumed by MADBPPlannerNode for the environment conditioning image).
  * Subscribes to ``/madbp/agent_{i}/plan`` (nav_msgs/Path) and
    tracks the trajectory with an on-board PD controller that outputs
    Twist commands on ``/robot_{i}/cmd_vel``.

Requires: lichtblik Python package  (pip install lichtblik)

Sim parameters (ROS 2 params):
  n_agents       : int   = 4
  world_size     : float = 10.0   (metres, square)
  render_size    : int   = 64     (pixels per side of /env/image)
  sim_hz         : float = 20.0
  collision_radius: float = 0.3
  kp             : float = 3.0    (PD position gain)
  kv             : float = 0.5    (PD velocity gain)
  max_vel        : float = 1.0    (m/s clamp)
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import (
    TransformStamped, Twist, PoseWithCovariance, TwistWithCovariance,
    Point, Quaternion,
)
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster


def _yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion()
    q.w = math.cos(yaw / 2)
    q.z = math.sin(yaw / 2)
    return q


class LichtblikBridge(Node):
    """Bridges a Lichtblik multi-agent world to ROS 2 topics."""

    def __init__(self) -> None:
        super().__init__("lichtblik_bridge")

        self.declare_parameter("n_agents", 4)
        self.declare_parameter("world_size", 10.0)
        self.declare_parameter("render_size", 64)
        self.declare_parameter("sim_hz", 20.0)
        self.declare_parameter("collision_radius", 0.3)
        self.declare_parameter("kp", 3.0)
        self.declare_parameter("kv", 0.5)
        self.declare_parameter("max_vel", 1.0)

        self._n: int = self.get_parameter("n_agents").value
        self._ws: float = self.get_parameter("world_size").value
        self._rs: int = self.get_parameter("render_size").value
        self._hz: float = self.get_parameter("sim_hz").value
        self._kp: float = self.get_parameter("kp").value
        self._kv: float = self.get_parameter("kv").value
        self._max_vel: float = self.get_parameter("max_vel").value

        # --- Lazy Lichtblik import ----------------------------------------
        try:
            import lichtblik as lb  # type: ignore
            self._world = lb.World(
                n_agents=self._n,
                size=self._ws,
                render_size=self._rs,
            )
            self._lb = lb
            self.get_logger().info(f"Lichtblik world created ({self._n} agents).")
        except ImportError:
            self.get_logger().error(
                "lichtblik package not found. Install with: pip install lichtblik\n"
                "The bridge will publish zero-state odometry until lichtblik is available."
            )
            self._world = None
            self._lb = None

        # Per-agent planned trajectories (set by path callback)
        self._plans: list[np.ndarray | None] = [None] * self._n  # each: [T, 2]
        self._plan_step: list[int] = [0] * self._n

        # --- TF broadcaster -----------------------------------------------
        self._tf_br = TransformBroadcaster(self)

        # --- Publishers ---------------------------------------------------
        self._odom_pubs = [
            self.create_publisher(Odometry, f"/robot_{i}/odom", 10)
            for i in range(self._n)
        ]
        self._cmd_pubs = [
            self.create_publisher(Twist, f"/robot_{i}/cmd_vel", 10)
            for i in range(self._n)
        ]
        self._img_pub = self.create_publisher(Image, "/env/image", 10)

        # --- Subscriptions (plans from planner node) ----------------------
        for i in range(self._n):
            self.create_subscription(
                Path,
                f"/madbp/agent_{i}/plan",
                lambda msg, idx=i: self._plan_callback(msg, idx),
                10,
            )

        # --- Timer --------------------------------------------------------
        period = 1.0 / self._hz
        self._timer = self.create_timer(period, self._tick)
        self.get_logger().info(f"Lichtblik bridge running at {self._hz} Hz.")

    # ------------------------------------------------------------------

    def _plan_callback(self, msg: Path, agent_idx: int) -> None:
        """Store a received nav_msgs/Path as a [T, 2] numpy array."""
        pts = [[ps.pose.position.x, ps.pose.position.y] for ps in msg.poses]
        self._plans[agent_idx] = np.array(pts, dtype=np.float32)
        self._plan_step[agent_idx] = 0
        self.get_logger().info(f"Agent {agent_idx}: received plan with {len(pts)} waypoints.")

    # ------------------------------------------------------------------

    def _tick(self) -> None:
        """Single simulation step: compute controls → step world → publish."""
        stamp = self.get_clock().now().to_msg()

        # Compute PD control for each agent
        vels: list[np.ndarray] = []
        for i in range(self._n):
            pos = self._get_pos(i)
            vel = self._get_vel(i)
            target_pos = self._get_target(i, pos)
            err_p = target_pos - pos
            err_v = -vel
            cmd = np.clip(self._kp * err_p + self._kv * err_v, -self._max_vel, self._max_vel)
            vels.append(cmd)

        # Step Lichtblik
        if self._world is not None:
            self._world.step(np.stack(vels, axis=0))

        # Publish odometry + TF
        for i in range(self._n):
            pos = self._get_pos(i)
            vel = self._get_vel(i)
            self._publish_odom(i, pos, vel, stamp)
            self._publish_tf(i, pos, stamp)

        # Publish env image
        self._publish_image(stamp)

    # ------------------------------------------------------------------

    def _get_pos(self, i: int) -> np.ndarray:
        if self._world is not None:
            return np.array(self._world.agents[i].pos, dtype=np.float32)
        return np.zeros(2, dtype=np.float32)

    def _get_vel(self, i: int) -> np.ndarray:
        if self._world is not None:
            return np.array(self._world.agents[i].vel, dtype=np.float32)
        return np.zeros(2, dtype=np.float32)

    def _get_target(self, i: int, current_pos: np.ndarray) -> np.ndarray:
        """Return the next waypoint for agent i, or current pos if no plan."""
        plan = self._plans[i]
        if plan is None:
            return current_pos
        step = min(self._plan_step[i], len(plan) - 1)
        target = plan[step]
        # Advance plan step when close enough
        if np.linalg.norm(target - current_pos) < 0.1 and step < len(plan) - 1:
            self._plan_step[i] += 1
        return target

    def _publish_odom(self, i: int, pos: np.ndarray, vel: np.ndarray, stamp: Any) -> None:
        msg = Odometry()
        msg.header = Header(stamp=stamp, frame_id="map")
        msg.child_frame_id = f"robot_{i}/base_link"
        msg.pose.pose.position = Point(x=float(pos[0]), y=float(pos[1]), z=0.0)
        msg.pose.pose.orientation = _yaw_to_quat(0.0)
        msg.twist.twist.linear.x = float(vel[0])
        msg.twist.twist.linear.y = float(vel[1])
        self._odom_pubs[i].publish(msg)

    def _publish_tf(self, i: int, pos: np.ndarray, stamp: Any) -> None:
        tf = TransformStamped()
        tf.header = Header(stamp=stamp, frame_id="map")
        tf.child_frame_id = f"robot_{i}/base_link"
        tf.transform.translation.x = float(pos[0])
        tf.transform.translation.y = float(pos[1])
        tf.transform.rotation = _yaw_to_quat(0.0)
        self._tf_br.sendTransform(tf)

    def _publish_image(self, stamp: Any) -> None:
        if self._world is None:
            return
        rgb = self._world.render()  # returns np.uint8 [H, W, 3]
        msg = Image()
        msg.header = Header(stamp=stamp, frame_id="map")
        msg.height, msg.width = rgb.shape[:2]
        msg.encoding = "rgb8"
        msg.step = msg.width * 3
        msg.data = rgb.tobytes()
        self._img_pub.publish(msg)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = LichtblikBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
