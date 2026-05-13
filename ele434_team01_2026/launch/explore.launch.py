#!/usr/bin/env python3

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    TimerAction,
    EmitEvent,
    SetEnvironmentVariable,
    IncludeLaunchDescription,
)
from launch.conditions import IfCondition
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def find_source_package_dir(package_name: str) -> Path:
    src_root = Path.home() / "ros2_ws" / "src"

    if src_root.exists():
        for package_xml in src_root.rglob("package.xml"):
            try:
                text = package_xml.read_text()
            except Exception:
                continue

            if f"<name>{package_name}</name>" in text:
                return package_xml.parent

    return Path(get_package_share_directory(package_name))


def generate_launch_description():
    package_name = "ele434_team01_2026"

    environment_arg = DeclareLaunchArgument(
        "environment",
        default_value="real",
        description="Use 'real' for the real robot or 'sim' for simulation.",
    )

    start_sim_arg = DeclareLaunchArgument(
        "start_sim",
        default_value="false",
        description="Set to 'true' to launch the simulation from this launch file.",
    )

    yaw_arg = DeclareLaunchArgument(
        "yaw",
        default_value="0.0",
        description="Initial yaw for simulation.",
    )

    run_duration_arg = DeclareLaunchArgument(
        "run_duration",
        default_value="90.0",
        description="Exploration duration in seconds.",
    )

    environment = LaunchConfiguration("environment")
    start_sim = LaunchConfiguration("start_sim")
    yaw = LaunchConfiguration("yaw")
    run_duration = LaunchConfiguration("run_duration")

    suppress_cartographer_info = SetEnvironmentVariable(
        name="GLOG_minloglevel",
        value="2",
    )

    normal_ros_logging = SetEnvironmentVariable(
        name="RCUTILS_LOGGING_MIN_SEVERITY",
        value="INFO",
    )

    src_pkg_dir = find_source_package_dir(package_name)
    maps_dir = src_pkg_dir / "maps"
    os.makedirs(maps_dir, exist_ok=True)

    map_output_base = str(maps_dir / "explore_map")

    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("tuos_task_sims"),
                "launch",
                "obstacle_avoidance.launch.py",
            )
        ),
        launch_arguments={
            "yaw": yaw,
            "use_sim_time": "true",
        }.items(),
        condition=IfCondition(start_sim),
    )

    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("tuos_tb3_tools"),
                "launch",
                "slam.launch.py",
            )
        ),
        launch_arguments={
            "environment": environment,
        }.items(),
    )

    explorer_node = TimerAction(
        period=5.0,
        actions=[
            Node(
                package=package_name,
                executable="slam_explorer.py",
                name="slam_explorer",
                output="screen",
                parameters=[
                    {
                        "run_duration": ParameterValue(
                            run_duration,
                            value_type=float,
                        ),
                        "max_speed": 0.24,
                        "cruise_speed": 0.22,
                        "turn_speed": 1.50,
                        "goal_tolerance": 0.15,

                        "debug_output_dir": str(maps_dir),
                        "save_debug_images": True,
                        "debug_image_prefix": "explore_debug",
                    }
                ],
            )
        ],
    )

    save_map = TimerAction(
        period=100.0,
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
        period=105.0,
        actions=[
            EmitEvent(
                event=Shutdown(
                    reason="Exploration complete and map saved."
                )
            )
        ],
    )

    return LaunchDescription([
        suppress_cartographer_info,
        normal_ros_logging,

        environment_arg,
        start_sim_arg,
        yaw_arg,
        run_duration_arg,

        sim_launch,
        slam_launch,
        explorer_node,
        save_map,
        shutdown_after_save,
    ])