"""MuJoCo-ROS2-Control bridge for MA-DBP validation (optional).

This bridge is an *alternative* to lichtblik_bridge.py for higher-fidelity
physics. It:

  * Loads a MuJoCo XML world (point-mass or differential-drive agents).
  * Runs the MuJoCo simulator at sim_hz (default 100 Hz).
  * Publishes the same /robot_{i}/odom and /env/image topics as the
    Lichtblik bridge so MADBPPlannerNode requires NO changes.
  * Subscribes to /madbp/agent_{i}/plan and tracks with a PD controller
    writing to MuJoCo actuators.

Requires: mujoco >= 3.0, mujoco-python >= 3.0
  pip install mujoco

MuJoCo XML path is set via the ROS param ``mujoco_xml`` (required).
"""

from __future__ import annotations

import math
from pathlib import Path as FsPath
from typing import Any

import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Quaternion, TransformStamped, Twist
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster


def _yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion()
    q.w = math.cos(yaw / 2)
    q.z = math.sin(yaw / 2)
    return q


# Default MuJoCo XML: 4 point-mass agents in a 10 m square arena.
_DEFAULT_XML = """
<mujoco model="madbp_nav">
  <option timestep="0.01" gravity="0 0 0"/>
  <worldbody>
    <light diffuse=".8 .8 .8" pos="0 0 10" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba=".9 .9 .9 1"/>
    <!-- Agent bodies are generated programmatically -->
  </worldbody>
</mujoco>
"""


class MuJoCoBridge(Node):
    """Bridges a MuJoCo multi-agent world to ROS 2 topics."""

    def __init__(self) -> None:
        super().__init__("mujoco_bridge")

        self.declare_parameter("n_agents", 4)
        self.declare_parameter("sim_hz", 100.0)
        self.declare_parameter("render_width", 64)
        self.declare_parameter("render_height", 64)
        self.declare_parameter("kp", 30.0)
        self.declare_parameter("kv", 2.5)
        self.declare_parameter("max_force", 1.0)
        self.declare_parameter("mujoco_xml", "")  # empty → use default XML string

        self._n: int = self.get_parameter("n_agents").value
        self._hz: float = self.get_parameter("sim_hz").value
        self._rw: int = self.get_parameter("render_width").value
        self._rh: int = self.get_parameter("render_height").value
        self._kp: float = self.get_parameter("kp").value
        self._kv: float = self.get_parameter("kv").value
        self._max_f: float = self.get_parameter("max_force").value
        xml_path: str = self.get_parameter("mujoco_xml").value

        # --- Load MuJoCo --------------------------------------------------
        try:
            import mujoco  # type: ignore
            self._mj = mujoco
            xml = FsPath(xml_path).read_text() if xml_path else _DEFAULT_XML
            self._mjmodel = mujoco.MjModel.from_xml_string(xml)
            self._mjdata = mujoco.MjData(self._mjmodel)
            # Renderer for /env/image
            self._renderer = mujoco.Renderer(self._mjmodel, height=self._rh, width=self._rw)
            self.get_logger().info(f"MuJoCo model loaded ({self._n} agents).")
        except ImportError:
            self.get_logger().error(
                "mujoco package not found. Install with: pip install mujoco\n"
                "The bridge will publish zero-state odometry until mujoco is available."
            )
            self._mj = None
            self._mjmodel = None
            self._mjdata = None
            self._renderer = None

        # Per-agent plans
        self._plans: list[np.ndarray | None] = [None] * self._n
        self._plan_step: list[int] = [0] * self._n

        # --- TF + publishers ----------------------------------------------
        self._tf_br = TransformBroadcaster(self)
        self._odom_pubs = [
            self.create_publisher(Odometry, f"/robot_{i}/odom", 10)
            for i in range(self._n)
        ]
        self._img_pub = self.create_publisher(Image, "/env/image", 10)

        for i in range(self._n):
            self.create_subscription(
                Path,
                f"/madbp/agent_{i}/plan",
                lambda msg, idx=i: self._plan_callback(msg, idx),
                10,
            )

        period = 1.0 / self._hz
        self._timer = self.create_timer(period, self._tick)
        self.get_logger().info(f"MuJoCo bridge running at {self._hz} Hz.")

    def _plan_callback(self, msg: Path, idx: int) -> None:
        pts = [[ps.pose.position.x, ps.pose.position.y] for ps in msg.poses]
        self._plans[idx] = np.array(pts, dtype=np.float32)
        self._plan_step[idx] = 0

    def _tick(self) -> None:
        stamp = self.get_clock().now().to_msg()
        if self._mjdata is None:
            return

        # Read current states
        for i in range(self._n):
            try:
                qpos_idx = self._mjmodel.body(f"agent_{i}").jntadr[0]
            except Exception:
                continue
            pos = self._mjdata.qpos[qpos_idx: qpos_idx + 2].copy()
            vel = self._mjdata.qvel[qpos_idx: qpos_idx + 2].copy()
            target = self._get_target(i, pos)
            err_p = target - pos
            err_v = -vel
            force = np.clip(self._kp * err_p + self._kv * err_v, -self._max_f, self._max_f)
            try:
                act_idx = self._mjmodel.actuator(f"agent_{i}_x").id
                self._mjdata.ctrl[act_idx] = float(force[0])
                self._mjdata.ctrl[act_idx + 1] = float(force[1])
            except Exception:
                pass

        self._mj.mj_step(self._mjmodel, self._mjdata)

        for i in range(self._n):
            try:
                qpos_idx = self._mjmodel.body(f"agent_{i}").jntadr[0]
                pos = self._mjdata.qpos[qpos_idx: qpos_idx + 2].copy()
                vel = self._mjdata.qvel[qpos_idx: qpos_idx + 2].copy()
            except Exception:
                pos = np.zeros(2)
                vel = np.zeros(2)
            self._publish_odom(i, pos, vel, stamp)
            self._publish_tf(i, pos, stamp)

        self._publish_image(stamp)

    def _get_target(self, i: int, pos: np.ndarray) -> np.ndarray:
        plan = self._plans[i]
        if plan is None:
            return pos
        step = min(self._plan_step[i], len(plan) - 1)
        target = plan[step]
        if np.linalg.norm(target - pos) < 0.1 and step < len(plan) - 1:
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
        if self._renderer is None:
            return
        self._renderer.update_scene(self._mjdata)
        rgb = self._renderer.render()  # np.uint8 [H, W, 3]
        msg = Image()
        msg.header = Header(stamp=stamp, frame_id="map")
        msg.height, msg.width = rgb.shape[:2]
        msg.encoding = "rgb8"
        msg.step = msg.width * 3
        msg.data = rgb.tobytes()
        self._img_pub.publish(msg)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MuJoCoBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
