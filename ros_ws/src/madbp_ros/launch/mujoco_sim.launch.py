"""Launch file: MA-DBP planner + MuJoCo bridge.

  ros2 launch madbp_ros mujoco_sim.launch.py \\
    checkpoint:=checkpoints/madbp_navigation_v2_n4_epoch_01999.pth \\
    n_agents:=4 \\
    mujoco_xml:=ros_ws/src/madbp_ros/assets/nav_arena.xml
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    pkg = FindPackageShare("madbp_ros")
    default_cfg = PathJoinSubstitution([pkg, "configs", "mujoco_nav.yaml"])

    return LaunchDescription([
        DeclareLaunchArgument("checkpoint", default_value="",
                              description="Path to MA-DBP .pth checkpoint"),
        DeclareLaunchArgument("n_agents", default_value="4",
                              description="Number of agents"),
        DeclareLaunchArgument("mujoco_xml", default_value="",
                              description="Path to MuJoCo XML world file (empty = built-in default)"),
        DeclareLaunchArgument("params_file", default_value=default_cfg,
                              description="ROS 2 params YAML"),

        Node(
            package="madbp_ros",
            executable="madbp_planner_node",
            name="madbp_planner",
            output="screen",
            parameters=[
                LaunchConfiguration("params_file"),
                {"checkpoint": LaunchConfiguration("checkpoint")},
            ],
        ),

        Node(
            package="madbp_ros",
            executable="mujoco_bridge",
            name="mujoco_bridge",
            output="screen",
            parameters=[
                LaunchConfiguration("params_file"),
                {"n_agents": LaunchConfiguration("n_agents"),
                 "mujoco_xml": LaunchConfiguration("mujoco_xml")},
            ],
        ),
    ])
