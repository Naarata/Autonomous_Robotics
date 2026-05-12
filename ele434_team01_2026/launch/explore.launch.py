#!/usr/bin/env python3

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    TimerAction,
    EmitEvent,
)
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_name = "ele434_team01_2026"  # <-- 반드시 팀 번호로 바꾸기

    environment_arg = DeclareLaunchArgument(
        "environment",
        default_value="real",
        description="Use 'real' for the real robot, or 'sim' for simulation.",
    )

    environment = LaunchConfiguration("environment")

    # The assignment requires maps/explore_map.png and maps/explore_map.yaml
    # at the root of the source package directory.
    src_pkg_dir = Path.home() / "ros2_ws" / "src" / package_name
    maps_dir = src_pkg_dir / "maps"
    os.makedirs(maps_dir, exist_ok=True)

    map_output_base = str(maps_dir / "explore_map")

    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("tuos_tb3_tools"),
                "launch",
                "slam.launch.py",
            ])
        ),
        launch_arguments={
            "environment": environment,
        }.items(),
    )

    explorer_node = Node(
        package=package_name,
        executable="slam_explorer.py",
        name="slam_explorer",
        output="screen",
        parameters=[
            {
                "run_duration": 90.0,
                "max_speed": 0.16,
                "max_angular_speed": 0.90,
                "goal_tolerance": 0.22,
            }
        ],
    )

    # Save the SLAM map shortly after the 90-second run.
    # --fmt png is used because the assignment asks specifically for explore_map.png.
    save_map = TimerAction(
        period=91.5,
        actions=[
            ExecuteProcess(
                cmd=[
                    "ros2",
                    "run",
                    "nav2_map_server",
                    "map_saver_cli",
                    "-t",
                    "/map",
                    "-f",
                    map_output_base,
                    "--fmt",
                    "png",
                ],
                output="screen",
            )
        ],
    )

    # Optional: shut down launch after map saving has had time to complete.
    shutdown_after_save = TimerAction(
        period=96.0,
        actions=[
            EmitEvent(event=Shutdown(reason="90-second exploration and map save complete."))
        ],
    )

    return LaunchDescription([
        environment_arg,
        slam_launch,
        explorer_node,
        save_map,
        shutdown_after_save,
    ])