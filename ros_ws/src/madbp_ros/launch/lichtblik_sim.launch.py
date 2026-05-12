"""Launch file: MA-DBP planner + Lichtblik bridge.

  ros2 launch madbp_ros lichtblik_sim.launch.py \\
    checkpoint:=checkpoints/madbp_navigation_v2_n4_epoch_01999.pth \\
    n_agents:=4
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    pkg = FindPackageShare("madbp_ros")
    default_cfg = PathJoinSubstitution([pkg, "configs", "lichtblik_nav.yaml"])

    return LaunchDescription([
        DeclareLaunchArgument("checkpoint", default_value="",
                              description="Path to MA-DBP .pth checkpoint"),
        DeclareLaunchArgument("n_agents", default_value="4",
                              description="Number of agents to deploy"),
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
            executable="lichtblik_bridge",
            name="lichtblik_bridge",
            output="screen",
            parameters=[
                LaunchConfiguration("params_file"),
                {"n_agents": LaunchConfiguration("n_agents")},
            ],
        ),
    ])
