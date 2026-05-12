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
    package_name = "ele434_team01_2026"


    environment_arg = DeclareLaunchArgument(
        "environment",
        default_value="real",
        description="real or sim",
    )


    environment = LaunchConfiguration("environment")


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


    explorer_node = TimerAction(
        period=3.0,
        actions=[
            Node(
                package=package_name,
                executable="slam_explorer.py",
                name="slam_explorer",
                output="screen",
                parameters=[
                    {
                        "run_duration": 90.0,
                        "max_speed": 0.17,
                        "cruise_speed": 0.15,
                        "corner_speed": 0.09,
                        "max_angular_speed": 0.95,
                        "goal_tolerance": 0.23,
                    }
                ],
            )
        ],
    )


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


    shutdown_after_save = TimerAction(
        period=96.0,
        actions=[
            EmitEvent(event=Shutdown(reason="Exploration complete and map saved."))
        ],
    )


    return LaunchDescription([
        environment_arg,
        slam_launch,
        explorer_node,
        save_map,
        shutdown_after_save,
    ])

