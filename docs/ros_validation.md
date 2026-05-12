# ROS 2 Validation Guide — MA-DBP on Real Systems

This document describes how to run the MA-DBP planner inside a ROS 2
environment using either **Lichtblik** (fast 2-D sim, recommended) or
**MuJoCo-ROS2-Control** (higher-fidelity physics, optional).

All ROS code lives in `ros_ws/src/madbp_ros/`.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Simulator (Lichtblik  OR  MuJoCo)                                  │
│  Bridge node publishes:                                             │
│    /env/image          → sensor_msgs/Image   (env top-down render)  │
│    /robot_i/odom       → nav_msgs/Odometry   (ground-truth poses)   │
│    TF: map → robot_i/base_link                                      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  /env/image
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  madbp_planner_node   (Action Server)                               │
│  Action:  madbp_interfaces/action/PlanMultiAgentPath                │
│    Goal : starts[n], goals[n], n_agents                             │
│    Result: success (bool)                                           │
│  Publishes: /madbp/agent_i/plan  → nav_msgs/Path                    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  /madbp/agent_i/plan
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Bridge PD controller  (inside the bridge node)                     │
│  Tracks each Path waypoint, writes cmd_vel / actuator forces        │
└─────────────────────────────────────────────────────────────────────┘
```

The planner node is **simulator-agnostic**: it only subscribes to
`/env/image` and publishes `nav_msgs/Path`. Swapping Lichtblik for
MuJoCo requires only changing which bridge you launch.

---

## Prerequisites

### ROS 2

ROS 2 Humble (Ubuntu 22.04) or ROS 2 Jazzy (Ubuntu 24.04):

```bash
# Ubuntu 22.04 / Humble
sudo apt install ros-humble-desktop ros-humble-launch-ros
source /opt/ros/humble/setup.bash
```

### Python dependencies

```bash
# Core MA-DBP package (from repo root)
pip install -e ".[dev]"

# Lichtblik (primary sim)
pip install lichtblik

# MuJoCo (optional, higher-fidelity)
pip install mujoco
```

### madbp_interfaces package

The action interface must be built first:

```bash
# Clone or create the interfaces package in ros_ws/src/
# (A separate madbp_interfaces package is required — see below.)
cd ros_ws
colcon build --packages-select madbp_interfaces
source install/setup.bash
```

#### Minimal madbp_interfaces action definition

Create `ros_ws/src/madbp_interfaces/action/PlanMultiAgentPath.action`:

```
# Goal
geometry_msgs/Point[] starts
geometry_msgs/Point[] goals
int32 n_agents
---
# Result
bool success
---
# Feedback  (empty for now)
```

---

## Build the madbp_ros package

```bash
cd ros_ws
colcon build --packages-select madbp_ros
source install/setup.bash
```

---

## Running with Lichtblik (recommended)

```bash
# Terminal 1: launch planner + sim bridge
ros2 launch madbp_ros lichtblik_sim.launch.py \\
  checkpoint:=checkpoints/madbp_navigation_v2_n4_epoch_01999.pth \\
  n_agents:=4

# Terminal 2: send a plan request manually (or use the ros_runner)
ros2 action send_goal /plan_multi_agent_path \\
  madbp_interfaces/action/PlanMultiAgentPath \\
  "{n_agents: 4,
    starts: [{x: -0.8, y: -0.8, z: 0}, {x: 0.8, y: -0.8, z: 0},
             {x: -0.8, y:  0.8, z: 0}, {x: 0.8, y:  0.8, z: 0}],
    goals:  [{x:  0.8, y:  0.8, z: 0}, {x: -0.8, y:  0.8, z: 0},
             {x:  0.8, y: -0.8, z: 0}, {x: -0.8, y: -0.8, z: 0}]}"
```

### Zero-shot scale-up test

Train on N=4, deploy on N=8:

```bash
ros2 launch madbp_ros lichtblik_sim.launch.py \\
  checkpoint:=checkpoints/madbp_navigation_v2_n4_epoch_01999.pth \\
  n_agents:=8   # ← larger than training team
```

The model's `max_agents=10` padding and axial-attention design handle
this without retraining.

---

## Running with MuJoCo (optional)

```bash
ros2 launch madbp_ros mujoco_sim.launch.py \\
  checkpoint:=checkpoints/madbp_navigation_v2_n4_epoch_01999.pth \\
  n_agents:=4 \\
  mujoco_xml:=ros_ws/src/madbp_ros/assets/nav_arena.xml
```

The MuJoCo bridge runs at 100 Hz (vs. Lichtblik's 20 Hz) and applies
physically-correct actuator dynamics. The planner node is identical in
both cases.

---

## Batch evaluation via `madbp-eval`

The `ROSRunner` in `ros_ws/src/madbp_ros/evaluation/ros_runner.py`
implements the same `RunnerProtocol` as `MADPRunner`, so you can use
the existing CLI:

```bash
madbp-eval --method ros \\
           --scenario navigation_v2 \\
           --n-agents 4 \\
           --seed 0 \\
           --n-episodes 20 \\
           --out results/ros_navigation_v2_n4.csv
```

Results land in the same CSV format as VMAS-based runs, enabling
direct comparison.

---

## Key topics reference

| Topic | Type | Direction | Description |
|---|---|---|---|
| `/env/image` | `sensor_msgs/Image` | Bridge → Planner | Top-down render (64×64 RGB) |
| `/robot_i/odom` | `nav_msgs/Odometry` | Bridge → All | Ground-truth pose + velocity |
| `/madbp/agent_i/plan` | `nav_msgs/Path` | Planner → Bridge | MA-DBP trajectory waypoints |
| `plan_multi_agent_path` | Action | Client → Planner | Trigger planning for a new episode |

---

## Parameter tuning

- **PD gains** (`kp`, `kv`): Start with Lichtblik defaults (3.0, 0.5). For
  MuJoCo point-mass agents use higher gains (30.0, 2.5) matching the
  original VMAS runner.
- **`max_step_size`**: Controls trajectory smoothness. Default 0.05 m/step
  works well for 20 Hz control. Reduce to 0.02 for slower, smoother
  motion on real hardware.
- **`cfg_scale`**: Set > 1.0 (e.g., 2.0) to sharpen goal-directed
  behaviour when using a model trained with CFG dropout > 0.
- **`full_horizon_length`**: Must be a multiple of `horizon` (default 8).
  Use 64–128 for navigation tasks.
