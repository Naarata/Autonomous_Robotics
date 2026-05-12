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
)
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PythonExpression

from launch_ros.actions import Node


def find_cartographer_config():
    """
    Find a Cartographer .lua config inside tuos_tb3_tools.

    This avoids including tuos_tb3_tools/slam.launch.py because that launch file
    starts RViz, and RViz's Cartographer Submaps display can spam /submap_query.
    """

    share_dir = Path(get_package_share_directory("tuos_tb3_tools"))

    candidates = list(share_dir.rglob("*.lua"))

    if not candidates:
        raise RuntimeError(
            f"No Cartographer .lua config found inside {share_dir}. "
            f"Run: find $(ros2 pkg prefix tuos_tb3_tools)/share/tuos_tb3_tools -name '*.lua'"
        )

    # Prefer likely Cartographer/TurtleBot3/Waffle configs.
    preferred_keywords = [
        "cartographer",
        "waffle",
        "turtlebot3",
        "tb3",
        "real",
        "sim",
    ]

    scored = []

    for path in candidates:
        name = path.name.lower()
        parent = str(path.parent).lower()

        score = 0
        for keyword in preferred_keywords:
            if keyword in name or keyword in parent:
                score += 1

        scored.append((score, path))

    scored.sort(key=lambda item: item[0], reverse=True)

    selected = scored[0][1]

    return str(selected.parent), selected.name


def generate_launch_description():
    package_name = "ele434_team01_2026"  # 네 팀 패키지 이름으로 유지/수정

    environment_arg = DeclareLaunchArgument(
        "environment",
        default_value="real",
        description="Use 'real' for the real robot or 'sim' for simulation.",
    )

    environment = LaunchConfiguration("environment")

    use_sim_time = PythonExpression([
        "'", environment, "' == 'sim'"
    ])

    src_pkg_dir = Path.home() / "ros2_ws" / "src" / package_name
    maps_dir = src_pkg_dir / "maps"
    os.makedirs(maps_dir, exist_ok=True)

    map_output_base = str(maps_dir / "explore_map")

    config_dir, config_basename = find_cartographer_config()

    # Suppress Cartographer glog terminal spam.
    suppress_cartographer_info = SetEnvironmentVariable(
        name="GLOG_minloglevel",
        value="2",
    )

    # Do not suppress our own explorer logs too much.
    # WARN means Cartographer INFO messages will be quieter, but explorer WARN still shows.
    suppress_ros_info = SetEnvironmentVariable(
        name="RCUTILS_LOGGING_MIN_SEVERITY",
        value="INFO",
    )

    cartographer_node = Node(
        package="cartographer_ros",
        executable="cartographer_node",
        name="cartographer_node",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
            }
        ],
        arguments=[
            "-configuration_directory",
            config_dir,
            "-configuration_basename",
            config_basename,
        ],
        remappings=[
            ("scan", "/scan"),
            ("odom", "/odom"),
        ],
    )

    occupancy_grid_node = Node(
        package="cartographer_ros",
        executable="cartographer_occupancy_grid_node",
        name="cartographer_occupancy_grid_node",
        output="log",
        parameters=[
            {
                "use_sim_time": use_sim_time,
            }
        ],
        arguments=[
            "-resolution",
            "0.05",
            "-publish_period_sec",
            "1.0",
        ],
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
        period=92.0,
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
        period=97.0,
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
        suppress_ros_info,
        environment_arg,
        cartographer_node,
        occupancy_grid_node,
        explorer_node,
        save_map,
        shutdown_after_save,
    ])