#!/usr/bin/env python3

import heapq
import json
import math
import os
import time
from enum import Enum
from pathlib import Path

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from rclpy.duration import Duration

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TwistStamped, Quaternion
from nav_msgs.msg import Odometry, OccupancyGrid

try:
    from tf2_ros import Buffer, TransformListener
except Exception:
    Buffer = None
    TransformListener = None


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def quaternion_to_yaw(orientation: Quaternion) -> float:
    x = orientation.x
    y = orientation.y
    z = orientation.z
    w = orientation.w

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)

    return math.atan2(t3, t4)


class ExplorerState(Enum):
    WAITING = 0
    MAP_PLAN = 1
    LOCAL_FALLBACK = 2
    SAFETY_TURN = 3
    RECOVERY_TURN = 4


class SlamExplorer(Node):

    def __init__(self):
        super().__init__("slam_explorer")

        # -------------------------------------------------
        # Parameters
        # -------------------------------------------------
        self.declare_parameter("run_duration", 90.0)
        self.declare_parameter("max_speed", 0.23)
        self.declare_parameter("cruise_speed", 0.21)
        self.declare_parameter("turn_speed", 1.35)
        self.declare_parameter("goal_tolerance", 0.22)

        self.run_duration = float(self.get_parameter("run_duration").value)
        self.max_speed = float(self.get_parameter("max_speed").value)
        self.cruise_speed = float(self.get_parameter("cruise_speed").value)
        self.turn_speed = float(self.get_parameter("turn_speed").value)
        self.goal_tolerance = float(self.get_parameter("goal_tolerance").value)

        # -------------------------------------------------
        # A* / costmap settings
        # -------------------------------------------------
        # The hard inflation radius is kept close to the previous working value.
        # Only the soft inflation radius is increased by about 5 cm so that A*
        # prefers paths that stay slightly farther away from walls without
        # blocking narrow gaps between thin walls and coloured posts.
        self.robot_radius = 0.135
        self.safety_margin = 0.035
        self.inflation_radius = self.robot_radius + self.safety_margin

        # Previous value was 0.27 m. This is increased by 0.05 m.
        self.soft_inflation_radius = 0.32

        self.replan_period = 0.75
        self.last_plan_time = 0.0

        self.path_cells = []
        self.path_world = []

        self.lookahead_open = 0.52
        self.lookahead_caution = 0.34

        self.occupied_threshold = 65
        self.unknown_cost = 2.1
        self.free_cost = 1.0

        # -------------------------------------------------
        # LiDAR safety settings
        # -------------------------------------------------
        # Hard stop distances are only slightly increased. The main clearance
        # improvement comes from the caution distances and side steering bias.
        self.center_emergency_distance = 0.17
        self.guard_emergency_distance = 0.20
        self.center_stop_distance = 0.28
        self.guard_stop_distance = 0.27
        self.side_stop_distance = 0.12

        # These caution thresholds are about 5 cm larger than the previous version.
        self.center_caution_distance = 0.53
        self.guard_caution_distance = 0.45
        self.wide_caution_distance = 0.35
        self.side_caution_distance = 0.22

        # Desired side clearance used by the continuous side steering bias.
        self.desired_side_clearance = 0.24
        self.side_bias_gain = 0.32
        self.wide_bias_gain = 0.18

        # ROS angular z: positive = left, negative = right.
        # The default escape direction is biased to the right.
        self.default_turn_direction = -1.0

        self.safety_until = 0.0
        self.safety_direction = self.default_turn_direction

        self.recovery_until = 0.0
        self.recovery_direction = self.default_turn_direction
        self.stuck_recovery_count = 0

        self.last_angular_cmd = 0.0
        self.angular_smoothing = 0.35

        # -------------------------------------------------
        # Debug saving
        # -------------------------------------------------
        self.declare_parameter("debug_output_dir", "")
        self.declare_parameter("save_debug_images", True)
        self.declare_parameter("debug_image_prefix", "explore_debug")
        self.declare_parameter("debug_image_scale", 8)

        self.debug_output_dir = str(self.get_parameter("debug_output_dir").value)
        self.save_debug_images_enabled = bool(
            self.get_parameter("save_debug_images").value
        )
        self.debug_image_prefix = str(
            self.get_parameter("debug_image_prefix").value
        )
        self.debug_image_scale = int(
            self.get_parameter("debug_image_scale").value
        )

        if self.debug_output_dir == "":
            self.debug_output_dir = str(Path.home() / "ros2_ws" / "debug_maps")

        os.makedirs(self.debug_output_dir, exist_ok=True)

        self.trajectory_world = []
        self.path_history_world = []
        self.debug_saved = False
        self.last_debug_pose_time = 0.0

        # -------------------------------------------------
        # ROS interfaces
        # -------------------------------------------------
        self.lidar_sub = self.create_subscription(
            LaserScan,
            "/scan",
            self.lidar_callback,
            10,
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
            10,
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            "/map",
            self.map_callback,
            10,
        )

        self.vel_pub = self.create_publisher(
            TwistStamped,
            "/cmd_vel",
            10,
        )

        if Buffer is not None and TransformListener is not None:
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
        else:
            self.tf_buffer = None
            self.tf_listener = None

        self.timer = self.create_timer(
            1.0 / 20.0,
            self.navigation_control,
        )

        # -------------------------------------------------
        # Robot state
        # -------------------------------------------------
        self.scan_data = None
        self.map_msg = None
        self.map_array = None
        self.cost_grid = None
        self.inflated_blocked = None

        self.have_odom = False
        self.initialised_odom = False

        self.x_offset = 0.0
        self.y_offset = 0.0
        self.yaw_offset = 0.0

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0

        self.have_initial_map_pose = False
        self.initial_map_x = 0.0
        self.initial_map_y = 0.0
        self.initial_map_yaw = 0.0

        self.start_time = None
        self.shutdown_requested = False

        self.state = ExplorerState.WAITING

        # -------------------------------------------------
        # Waypoints
        # -------------------------------------------------
        # These points are slightly inside the arena perimeter. They provide
        # coverage of the numbered outer zones without intentionally hugging
        # the outer arena wall.
        self.waypoints = [
            (1.20, 1.20),
            (1.20, 0.40),
            (1.20, -0.40),
            (1.20, -1.20),
            (0.40, -1.20),
            (-0.40, -1.20),
            (-1.20, -1.20),
            (-1.20, -0.40),
            (-1.20, 0.40),
            (-1.20, 1.20),
            (-0.40, 1.20),
            (0.40, 1.20),
        ]

        self.current_waypoint = 0
        self.completed_laps = 0

        self.last_goal_distance = None
        self.last_progress_time = time.time()

        self.get_logger().info(
            "Gap-friendly A* slam_explorer initialised with extra wall clearance."
        )

    # -------------------------------------------------
    # ROS callbacks
    # -------------------------------------------------
    def lidar_callback(self, msg: LaserScan):
        self.scan_data = msg

    def odom_callback(self, msg: Odometry):
        pose = msg.pose.pose
        yaw = quaternion_to_yaw(pose.orientation)

        if not self.initialised_odom:
            self.x_offset = pose.position.x
            self.y_offset = pose.position.y
            self.yaw_offset = yaw
            self.initialised_odom = True

            self.get_logger().info(
                f"Initial odom stored: "
                f"x={self.x_offset:.2f}, "
                f"y={self.y_offset:.2f}, "
                f"yaw={self.yaw_offset:.2f}"
            )

        dx = pose.position.x - self.x_offset
        dy = pose.position.y - self.y_offset

        c = math.cos(-self.yaw_offset)
        s = math.sin(-self.yaw_offset)

        self.current_x = c * dx - s * dy
        self.current_y = s * dx + c * dy
        self.current_yaw = wrap_angle(yaw - self.yaw_offset)

        self.have_odom = True

    def map_callback(self, msg: OccupancyGrid):
        self.map_msg = msg

        try:
            self.map_array = np.array(
                msg.data,
                dtype=np.int16,
            ).reshape((msg.info.height, msg.info.width))

            self.build_costmap()

        except Exception as exc:
            self.get_logger().warn(f"Failed to process map: {exc}")
            self.map_array = None
            self.cost_grid = None
            self.inflated_blocked = None

    # -------------------------------------------------
    # TF / pose helpers
    # -------------------------------------------------
    def get_robot_map_pose(self):
        if self.tf_buffer is None:
            return None

        for base_frame in ["base_footprint", "base_link"]:
            try:
                tf = self.tf_buffer.lookup_transform(
                    "map",
                    base_frame,
                    Time(),
                    timeout=Duration(seconds=0.03),
                )

                t = tf.transform.translation
                q = tf.transform.rotation
                yaw = quaternion_to_yaw(q)

                return float(t.x), float(t.y), float(yaw)

            except Exception:
                continue

        return None

    def get_planning_pose(self):
        tf_pose = self.get_robot_map_pose()

        if tf_pose is not None:
            x, y, yaw = tf_pose

            if not self.have_initial_map_pose:
                self.initial_map_x = x
                self.initial_map_y = y
                self.initial_map_yaw = yaw
                self.have_initial_map_pose = True

                self.get_logger().info(
                    f"Initial map pose stored: "
                    f"x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}"
                )

            return x, y, yaw, True

        if not self.have_initial_map_pose:
            self.initial_map_x = 0.0
            self.initial_map_y = 0.0
            self.initial_map_yaw = 0.0
            self.have_initial_map_pose = True

        return self.current_x, self.current_y, self.current_yaw, False

    def waypoint_to_map(self, local_x: float, local_y: float):
        c = math.cos(self.initial_map_yaw)
        s = math.sin(self.initial_map_yaw)

        mx = self.initial_map_x + c * local_x - s * local_y
        my = self.initial_map_y + s * local_x + c * local_y

        return mx, my

    # -------------------------------------------------
    # Costmap helpers
    # -------------------------------------------------
    def world_to_map_cell(self, x: float, y: float):
        if self.map_msg is None:
            return None

        info = self.map_msg.info
        res = info.resolution
        ox = info.origin.position.x
        oy = info.origin.position.y

        mx = int((x - ox) / res)
        my = int((y - oy) / res)

        if mx < 0 or my < 0 or mx >= info.width or my >= info.height:
            return None

        return mx, my

    def map_cell_to_world(self, mx: int, my: int):
        info = self.map_msg.info
        res = info.resolution
        ox = info.origin.position.x
        oy = info.origin.position.y

        x = ox + (mx + 0.5) * res
        y = oy + (my + 0.5) * res

        return x, y

    def build_costmap(self):
        if self.map_msg is None or self.map_array is None:
            return

        info = self.map_msg.info
        res = info.resolution

        occupied = self.map_array >= self.occupied_threshold

        hard_cells = max(1, int(math.ceil(self.inflation_radius / res)))
        soft_cells = max(
            hard_cells + 1,
            int(math.ceil(self.soft_inflation_radius / res)),
        )

        hard_blocked = occupied.copy()
        penalty = np.zeros_like(self.map_array, dtype=np.float32)

        ys, xs = np.where(occupied)
        height, width = occupied.shape

        for dy in range(-soft_cells, soft_cells + 1):
            for dx in range(-soft_cells, soft_cells + 1):
                dist_cells = math.hypot(dx, dy)

                if dist_cells > soft_cells:
                    continue

                nx = xs + dx
                ny = ys + dy

                valid = (
                    (nx >= 0)
                    & (ny >= 0)
                    & (nx < width)
                    & (ny < height)
                )

                if not np.any(valid):
                    continue

                nx_valid = nx[valid]
                ny_valid = ny[valid]

                if dist_cells <= hard_cells:
                    hard_blocked[ny_valid, nx_valid] = True
                else:
                    weight = 1.0 - (
                        (dist_cells - hard_cells)
                        / max(1.0, soft_cells - hard_cells)
                    )

                    # The soft penalty is increased so the planned path prefers
                    # a slightly more central line in corridors.
                    add_cost = 2.8 * max(0.0, weight)
                    penalty[ny_valid, nx_valid] = np.maximum(
                        penalty[ny_valid, nx_valid],
                        add_cost,
                    )

        cost_grid = np.ones_like(self.map_array, dtype=np.float32)
        cost_grid[self.map_array < 0] = self.unknown_cost
        cost_grid[self.map_array >= 0] = self.free_cost
        cost_grid += penalty
        cost_grid[hard_blocked] = np.inf

        self.cost_grid = cost_grid
        self.inflated_blocked = hard_blocked

    def is_cell_traversable(self, cell):
        if self.cost_grid is None:
            return False

        mx, my = cell
        height, width = self.cost_grid.shape

        if mx < 0 or my < 0 or mx >= width or my >= height:
            return False

        return np.isfinite(self.cost_grid[my, mx])

    def nearest_traversable_cell(self, cell, max_radius=12):
        if self.cost_grid is None:
            return None

        mx0, my0 = cell
        height, width = self.cost_grid.shape

        if mx0 < 0 or my0 < 0 or mx0 >= width or my0 >= height:
            return None

        if self.is_cell_traversable(cell):
            return cell

        best = None
        best_dist = 1e9

        for radius in range(1, max_radius + 1):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if abs(dx) != radius and abs(dy) != radius:
                        continue

                    mx = mx0 + dx
                    my = my0 + dy

                    if self.is_cell_traversable((mx, my)):
                        dist = dx * dx + dy * dy

                        if dist < best_dist:
                            best_dist = dist
                            best = (mx, my)

            if best is not None:
                return best

        return None

    # -------------------------------------------------
    # A* path planning
    # -------------------------------------------------
    def astar(self, start_cell, goal_cell, max_expansions=16000):
        if self.cost_grid is None:
            return []

        start = self.nearest_traversable_cell(start_cell)
        goal = self.nearest_traversable_cell(goal_cell)

        if start is None or goal is None:
            return []

        if start == goal:
            return [start]

        height, width = self.cost_grid.shape

        neighbours = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0)),
        ]

        def heuristic(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        open_heap = []
        heapq.heappush(open_heap, (0.0, start))

        came_from = {}
        g_score = {start: 0.0}
        closed = set()
        expansions = 0

        while open_heap and expansions < max_expansions:
            _, current = heapq.heappop(open_heap)

            if current in closed:
                continue

            if current == goal:
                return self.reconstruct_path(came_from, current)

            closed.add(current)
            expansions += 1

            cx, cy = current

            for dx, dy, move_cost in neighbours:
                nx = cx + dx
                ny = cy + dy

                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue

                neighbour = (nx, ny)

                if not self.is_cell_traversable(neighbour):
                    continue

                cell_cost = float(self.cost_grid[ny, nx])
                tentative_g = g_score[current] + move_cost * cell_cost

                if tentative_g < g_score.get(neighbour, float("inf")):
                    came_from[neighbour] = current
                    g_score[neighbour] = tentative_g
                    f = tentative_g + heuristic(neighbour, goal)
                    heapq.heappush(open_heap, (f, neighbour))

        return []

    def reconstruct_path(self, came_from, current):
        path = [current]

        while current in came_from:
            current = came_from[current]
            path.append(current)

        path.reverse()
        return path

    def plan_path_to_waypoint(self, robot_x, robot_y):
        if self.map_msg is None or self.cost_grid is None:
            return False

        goal_local = self.waypoints[self.current_waypoint]
        goal_map = self.waypoint_to_map(goal_local[0], goal_local[1])

        start_cell = self.world_to_map_cell(robot_x, robot_y)
        goal_cell = self.world_to_map_cell(goal_map[0], goal_map[1])

        if start_cell is None or goal_cell is None:
            self.path_cells = []
            self.path_world = []
            return False

        path_cells = self.astar(start_cell, goal_cell)

        if not path_cells:
            self.path_cells = []
            self.path_world = []
            return False

        self.path_cells = path_cells
        self.path_world = [
            self.map_cell_to_world(mx, my)
            for mx, my in path_cells
        ]

        if self.save_debug_images_enabled:
            self.path_history_world.append(list(self.path_world))

            if len(self.path_history_world) > 160:
                self.path_history_world = self.path_history_world[-160:]

        return True

    def choose_lookahead_point(self, robot_x, robot_y, lookahead_distance):
        if not self.path_world:
            return None

        closest_idx = 0
        closest_dist = float("inf")

        for i, (px, py) in enumerate(self.path_world):
            d = math.hypot(px - robot_x, py - robot_y)

            if d < closest_dist:
                closest_dist = d
                closest_idx = i

        for i in range(closest_idx, len(self.path_world)):
            px, py = self.path_world[i]
            d = math.hypot(px - robot_x, py - robot_y)

            if d >= lookahead_distance:
                return px, py

        return self.path_world[-1]

    # -------------------------------------------------
    # LiDAR processing
    # -------------------------------------------------
    def get_lidar_sectors(self):
        if self.scan_data is None:
            return None

        scan = self.scan_data
        ranges = np.array(scan.ranges, dtype=np.float32)

        safe_max = 3.5
        range_min = max(scan.range_min, 0.08)
        range_max = scan.range_max if scan.range_max > 0.0 else safe_max
        range_max = min(range_max, safe_max)

        valid = (
            np.isfinite(ranges)
            & (ranges >= range_min)
            & (ranges <= range_max)
        )

        ranges = np.where(valid, ranges, safe_max)

        indices = np.arange(len(ranges), dtype=np.float32)
        angles = scan.angle_min + indices * scan.angle_increment
        angles = np.arctan2(np.sin(angles), np.cos(angles))

        def robust_distance(mask, percentile=15.0):
            values = ranges[mask]
            if values.size == 0:
                return safe_max
            return float(np.percentile(values, percentile))

        def min_with_angle(mask):
            values = ranges[mask]
            value_angles = angles[mask]

            if values.size == 0:
                return safe_max, 0.0

            idx = int(np.argmin(values))
            return float(values[idx]), float(value_angles[idx])

        def min_distance(mask):
            values = ranges[mask]
            if values.size == 0:
                return safe_max
            return float(np.min(values))

        center_mask = (angles > -0.28) & (angles < 0.28)
        guard_mask = (angles > -0.58) & (angles < 0.58)
        wide_mask = (angles > -1.20) & (angles < 1.20)

        left_mask = (angles >= 0.45) & (angles < 1.30)
        right_mask = (angles <= -0.45) & (angles > -1.30)

        side_left_mask = (angles >= 1.00) & (angles < 1.57)
        side_right_mask = (angles <= -1.00) & (angles > -1.57)

        center_min, center_angle = min_with_angle(center_mask)
        guard_min, guard_angle = min_with_angle(guard_mask)
        wide_min, wide_angle = min_with_angle(wide_mask)

        return {
            "center": robust_distance(center_mask),
            "center_min": center_min,
            "center_angle": center_angle,
            "guard": robust_distance(guard_mask),
            "guard_min": guard_min,
            "guard_angle": guard_angle,
            "wide_min": wide_min,
            "wide_angle": wide_angle,
            "left": robust_distance(left_mask),
            "right": robust_distance(right_mask),
            "side_left_min": min_distance(side_left_mask),
            "side_right_min": min_distance(side_right_mask),
            "ranges": ranges,
            "angles": angles,
        }

    # -------------------------------------------------
    # Safety and local planning
    # -------------------------------------------------
    def choose_escape_direction(
        self,
        dist_left,
        dist_right,
        obstacle_angle=0.0,
        side_left_min=3.5,
        side_right_min=3.5,
    ):
        if side_left_min < self.side_stop_distance:
            return -1.0

        if side_right_min < self.side_stop_distance:
            return 1.0

        if obstacle_angle > 0.08:
            return -1.0

        if obstacle_angle < -0.08:
            return 1.0

        if dist_left - dist_right > 0.18:
            return 1.0

        if dist_right - dist_left > 0.18:
            return -1.0

        return self.default_turn_direction

    def has_hard_contact_risk(self, sectors):
        center = sectors["center"]
        center_min = sectors["center_min"]
        guard_min = sectors["guard_min"]
        guard_angle = sectors["guard_angle"]
        side_left_min = sectors["side_left_min"]
        side_right_min = sectors["side_right_min"]

        if center_min < self.center_emergency_distance:
            return True, "CENTER_EMERGENCY", sectors["center_angle"]

        if guard_min < self.guard_emergency_distance and abs(guard_angle) < 0.50:
            return True, "GUARD_EMERGENCY", guard_angle

        if center < self.center_stop_distance and guard_min < self.guard_stop_distance:
            return True, "FRONT_STOP", guard_angle

        if side_left_min < self.side_stop_distance:
            return True, "LEFT_SCRAPE", 1.2

        if side_right_min < self.side_stop_distance:
            return True, "RIGHT_SCRAPE", -1.2

        return False, "", 0.0

    def is_caution(self, sectors):
        if sectors["center"] < self.center_caution_distance:
            return True

        if sectors["guard_min"] < self.guard_caution_distance:
            return True

        if sectors["wide_min"] < self.wide_caution_distance:
            return True

        if sectors["side_left_min"] < self.side_caution_distance:
            return True

        if sectors["side_right_min"] < self.side_caution_distance:
            return True

        return False

    def wall_clearance_bias(self, sectors):
        bias = 0.0

        side_left = sectors["side_left_min"]
        side_right = sectors["side_right_min"]
        wide_min = sectors["wide_min"]
        wide_angle = sectors["wide_angle"]

        if side_left < self.desired_side_clearance:
            error = self.desired_side_clearance - side_left
            bias -= self.side_bias_gain * (error / self.desired_side_clearance)

        if side_right < self.desired_side_clearance:
            error = self.desired_side_clearance - side_right
            bias += self.side_bias_gain * (error / self.desired_side_clearance)

        if wide_min < self.wide_caution_distance:
            error = self.wide_caution_distance - wide_min
            wide_push = self.wide_bias_gain * (error / self.wide_caution_distance)

            if wide_angle > 0.20:
                bias -= wide_push
            elif wide_angle < -0.20:
                bias += wide_push

        return max(-0.35, min(0.35, bias))

    def fallback_local_angle(self, sectors, goal_angle_local):
        ranges = sectors["ranges"]
        angles = sectors["angles"]

        candidate_angles = np.linspace(-0.95, 0.95, 39)

        best_angle = 0.0
        best_score = -1e9

        for candidate in candidate_angles:
            angular_distance = np.abs(
                np.arctan2(
                    np.sin(angles - candidate),
                    np.cos(angles - candidate),
                )
            )

            mask = angular_distance < 0.22
            values = ranges[mask]

            if values.size == 0:
                clearance = 3.5
            else:
                clearance = float(np.percentile(values, 18.0))

            if clearance < 0.30:
                continue

            goal_penalty = abs(wrap_angle(candidate - goal_angle_local))
            score = 2.2 * min(clearance, 1.6)
            score -= 1.8 * goal_penalty
            score -= 0.16 * abs(candidate)

            if candidate < -0.05:
                score += 0.08

            if score > best_score:
                best_score = score
                best_angle = float(candidate)

        return best_angle

    def speed_from_clearance(self, sectors, heading_error):
        center = sectors["center"]
        guard_min = sectors["guard_min"]
        wide_min = sectors["wide_min"]
        side_left_min = sectors["side_left_min"]
        side_right_min = sectors["side_right_min"]

        abs_heading = abs(heading_error)

        if center < self.center_stop_distance and guard_min < self.guard_stop_distance:
            return 0.0

        if abs_heading > 1.15:
            return 0.0

        if abs_heading > 0.80:
            return 0.06

        if abs_heading > 0.50:
            return 0.11

        if side_left_min < self.side_caution_distance:
            return 0.09

        if side_right_min < self.side_caution_distance:
            return 0.09

        if guard_min < self.guard_caution_distance:
            return 0.09

        if wide_min < self.wide_caution_distance:
            return 0.10

        if center < 0.55:
            return 0.11

        if center < 0.75:
            return 0.15

        return self.cruise_speed

    # -------------------------------------------------
    # Motion commands
    # -------------------------------------------------
    def publish_cmd(self, linear_x: float, angular_z: float, smooth=True):
        linear_x = max(0.0, min(self.max_speed, linear_x))

        angular_z = max(
            -self.turn_speed,
            min(self.turn_speed, angular_z),
        )

        if smooth:
            angular_z = (
                self.angular_smoothing * angular_z
                + (1.0 - self.angular_smoothing) * self.last_angular_cmd
            )

        self.last_angular_cmd = angular_z

        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"

        msg.twist.linear.x = float(linear_x)
        msg.twist.angular.z = float(angular_z)

        self.vel_pub.publish(msg)

        if self.start_time is None and (
            abs(linear_x) > 1e-3 or abs(angular_z) > 1e-3
        ):
            self.start_time = time.time()
            self.get_logger().info("90-second exploration timer started.")

    def stop_robot(self):
        for _ in range(6):
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            msg.twist.linear.x = 0.0
            msg.twist.angular.z = 0.0
            self.vel_pub.publish(msg)
            time.sleep(0.02)

    # -------------------------------------------------
    # Debug image helpers
    # -------------------------------------------------
    def record_debug_pose(self, robot_x: float, robot_y: float):
        if not self.save_debug_images_enabled:
            return

        now = time.time()

        if now - self.last_debug_pose_time < 0.10:
            return

        self.last_debug_pose_time = now
        self.trajectory_world.append((float(robot_x), float(robot_y)))

        if len(self.trajectory_world) > 5000:
            self.trajectory_world = self.trajectory_world[-5000:]

    def world_to_pixel_for_debug(self, x: float, y: float):
        if self.map_msg is None:
            return None

        info = self.map_msg.info
        res = info.resolution
        ox = info.origin.position.x
        oy = info.origin.position.y

        mx = int((x - ox) / res)
        my = int((y - oy) / res)

        if mx < 0 or my < 0 or mx >= info.width or my >= info.height:
            return None

        return mx, info.height - 1 - my

    def upscale_image(self, img):
        scale = max(1, int(self.debug_image_scale))
        if scale == 1:
            return img

        if img.ndim == 2:
            return np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)

        return np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)

    def save_gray_image(self, image_array, path: Path):
        try:
            from PIL import Image
            img = Image.fromarray(image_array.astype(np.uint8), mode="L")
            img.save(path)
            return str(path)
        except Exception:
            pass

        try:
            import cv2
            cv2.imwrite(str(path), image_array.astype(np.uint8))
            return str(path)
        except Exception:
            pass

        pgm_path = path.with_suffix(".pgm")
        h, w = image_array.shape

        with open(pgm_path, "wb") as f:
            f.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
            f.write(image_array.astype(np.uint8).tobytes())

        return str(pgm_path)

    def save_rgb_image(self, image_array, path: Path):
        try:
            from PIL import Image
            img = Image.fromarray(image_array.astype(np.uint8), mode="RGB")
            img.save(path)
            return str(path)
        except Exception:
            pass

        try:
            import cv2
            bgr = image_array[:, :, ::-1].astype(np.uint8)
            cv2.imwrite(str(path), bgr)
            return str(path)
        except Exception:
            pass

        ppm_path = path.with_suffix(".ppm")
        h, w, _ = image_array.shape

        with open(ppm_path, "wb") as f:
            f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
            f.write(image_array.astype(np.uint8).tobytes())

        return str(ppm_path)

    def draw_line_on_rgb(self, img, p0, p1, color, thickness=1):
        x0, y0 = p0
        x1, y1 = p1

        h, w, _ = img.shape

        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)

        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1

        err = dx + dy
        x = x0
        y = y0

        while True:
            for ty in range(-thickness, thickness + 1):
                for tx in range(-thickness, thickness + 1):
                    xx = x + tx
                    yy = y + ty

                    if 0 <= xx < w and 0 <= yy < h:
                        img[yy, xx, :] = color

            if x == x1 and y == y1:
                break

            e2 = 2 * err

            if e2 >= dy:
                err += dy
                x += sx

            if e2 <= dx:
                err += dx
                y += sy

    def draw_polyline_on_rgb(self, img, points, color, thickness=1):
        pixels = []

        for x, y in points:
            pix = self.world_to_pixel_for_debug(x, y)
            if pix is not None:
                pixels.append(pix)

        if len(pixels) < 2:
            return

        for i in range(len(pixels) - 1):
            self.draw_line_on_rgb(
                img,
                pixels[i],
                pixels[i + 1],
                color,
                thickness,
            )

    def draw_circle_on_rgb(self, img, center, radius, color):
        h, w, _ = img.shape
        cx, cy = center

        for y in range(cy - radius, cy + radius + 1):
            for x in range(cx - radius, cx + radius + 1):
                if 0 <= x < w and 0 <= y < h:
                    if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                        img[y, x, :] = color

    def save_debug_outputs(self):
        if self.debug_saved:
            return

        if not self.save_debug_images_enabled:
            return

        if self.map_msg is None or self.map_array is None:
            self.get_logger().warn("Debug save skipped: no map available.")
            return

        self.debug_saved = True

        output_dir = Path(self.debug_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        raw_map_path = output_dir / f"{self.debug_image_prefix}_raw_map.png"
        overlay_path = output_dir / f"{self.debug_image_prefix}_overlay.png"
        overlay_hd_path = output_dir / f"{self.debug_image_prefix}_overlay_hd.png"
        json_path = output_dir / f"{self.debug_image_prefix}_paths.json"

        try:
            data = self.map_array
            height, width = data.shape

            img_array = np.zeros((height, width), dtype=np.uint8)
            img_array[data < 0] = 205
            img_array[(data >= 0) & (data < self.occupied_threshold)] = 254
            img_array[data >= self.occupied_threshold] = 0

            img_array = np.flipud(img_array)

            saved_raw = self.save_gray_image(
                self.upscale_image(img_array),
                raw_map_path,
            )

            overlay = np.stack([img_array, img_array, img_array], axis=-1)

            for path in self.path_history_world:
                self.draw_polyline_on_rgb(
                    overlay,
                    path,
                    color=(40, 120, 255),
                    thickness=1,
                )

            if self.path_world:
                self.draw_polyline_on_rgb(
                    overlay,
                    self.path_world,
                    color=(0, 220, 255),
                    thickness=2,
                )

            self.draw_polyline_on_rgb(
                overlay,
                self.trajectory_world,
                color=(255, 40, 40),
                thickness=2,
            )

            for wp_x, wp_y in self.waypoints:
                wx, wy = self.waypoint_to_map(wp_x, wp_y)
                pix = self.world_to_pixel_for_debug(wx, wy)

                if pix is not None:
                    self.draw_circle_on_rgb(
                        overlay,
                        pix,
                        radius=3,
                        color=(0, 220, 0),
                    )

            if self.trajectory_world:
                lx, ly = self.trajectory_world[-1]
                pix = self.world_to_pixel_for_debug(lx, ly)

                if pix is not None:
                    self.draw_circle_on_rgb(
                        overlay,
                        pix,
                        radius=4,
                        color=(255, 230, 0),
                    )

            saved_overlay = self.save_rgb_image(overlay, overlay_path)
            saved_overlay_hd = self.save_rgb_image(
                self.upscale_image(overlay),
                overlay_hd_path,
            )

            debug_data = {
                "map": {
                    "resolution": float(self.map_msg.info.resolution),
                    "width": int(self.map_msg.info.width),
                    "height": int(self.map_msg.info.height),
                    "origin": {
                        "x": float(self.map_msg.info.origin.position.x),
                        "y": float(self.map_msg.info.origin.position.y),
                    },
                },
                "settings": {
                    "robot_radius": self.robot_radius,
                    "safety_margin": self.safety_margin,
                    "inflation_radius": self.inflation_radius,
                    "soft_inflation_radius": self.soft_inflation_radius,
                    "center_stop_distance": self.center_stop_distance,
                    "guard_stop_distance": self.guard_stop_distance,
                    "side_stop_distance": self.side_stop_distance,
                    "center_caution_distance": self.center_caution_distance,
                    "guard_caution_distance": self.guard_caution_distance,
                    "wide_caution_distance": self.wide_caution_distance,
                    "side_caution_distance": self.side_caution_distance,
                    "desired_side_clearance": self.desired_side_clearance,
                    "debug_image_scale": self.debug_image_scale,
                },
                "waypoints_local": [
                    {"x": float(x), "y": float(y)}
                    for x, y in self.waypoints
                ],
                "trajectory_world": [
                    {"x": float(x), "y": float(y)}
                    for x, y in self.trajectory_world
                ],
                "path_history_world": [
                    [
                        {"x": float(x), "y": float(y)}
                        for x, y in path
                    ]
                    for path in self.path_history_world
                ],
                "current_path_world": [
                    {"x": float(x), "y": float(y)}
                    for x, y in self.path_world
                ],
            }

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(debug_data, f, indent=2)

            self.get_logger().info(f"Saved debug raw map: {saved_raw}")
            self.get_logger().info(f"Saved debug overlay: {saved_overlay}")
            self.get_logger().info(f"Saved debug overlay HD: {saved_overlay_hd}")
            self.get_logger().info(f"Saved debug data: {json_path}")

        except Exception as exc:
            self.get_logger().warn(
                f"Failed to save debug images/data: {exc}"
            )

    # -------------------------------------------------
    # Main navigation loop
    # -------------------------------------------------
    def navigation_control(self):
        if self.shutdown_requested:
            return

        if not self.have_odom or self.scan_data is None:
            return

        if self.start_time is not None:
            elapsed = time.time() - self.start_time

            if elapsed >= self.run_duration:
                self.get_logger().info("Run duration complete. Stopping robot.")
                self.shutdown_requested = True
                self.stop_robot()
                self.save_debug_outputs()
                rclpy.shutdown()
                return

        sectors = self.get_lidar_sectors()
        if sectors is None:
            return

        now = time.time()

        robot_x, robot_y, robot_yaw, using_tf = self.get_planning_pose()
        self.record_debug_pose(robot_x, robot_y)

        dist_center = sectors["center"]
        dist_left = sectors["left"]
        dist_right = sectors["right"]
        guard_min = sectors["guard_min"]
        guard_angle = sectors["guard_angle"]
        wide_min = sectors["wide_min"]
        wide_angle = sectors["wide_angle"]
        side_left_min = sectors["side_left_min"]
        side_right_min = sectors["side_right_min"]

        # -------------------------------------------------
        # Safety turn lock
        # -------------------------------------------------
        if now < self.safety_until:
            self.state = ExplorerState.SAFETY_TURN
            self.publish_cmd(
                0.0,
                self.safety_direction * 0.85,
                smooth=False,
            )
            return

        # -------------------------------------------------
        # Hard contact risk check
        # -------------------------------------------------
        hard_risk, reason, risk_angle = self.has_hard_contact_risk(sectors)

        if hard_risk:
            turn_direction = self.choose_escape_direction(
                dist_left,
                dist_right,
                obstacle_angle=risk_angle,
                side_left_min=side_left_min,
                side_right_min=side_right_min,
            )

            self.safety_until = now + 0.55
            self.safety_direction = turn_direction
            self.path_cells = []
            self.path_world = []

            self.get_logger().warn(
                f"{reason}: center={dist_center:.2f}, "
                f"guard={guard_min:.2f}@{guard_angle:.2f}, "
                f"wide={wide_min:.2f}@{wide_angle:.2f}, "
                f"SL={side_left_min:.2f}, SR={side_right_min:.2f}, "
                f"turn={turn_direction:.0f}",
                throttle_duration_sec=0.6,
            )

            self.publish_cmd(
                0.0,
                turn_direction * 0.85,
                smooth=False,
            )
            return

        # -------------------------------------------------
        # Waypoint check
        # -------------------------------------------------
        goal_local_x, goal_local_y = self.waypoints[self.current_waypoint]

        local_dx = goal_local_x - self.current_x
        local_dy = goal_local_y - self.current_y

        distance_to_goal = math.hypot(local_dx, local_dy)

        if distance_to_goal < self.goal_tolerance:
            self.get_logger().info(
                f"Reached waypoint {self.current_waypoint + 1}/12"
            )

            self.current_waypoint += 1

            if self.current_waypoint >= len(self.waypoints):
                self.current_waypoint = 0
                self.completed_laps += 1
                self.get_logger().info(
                    f"Completed waypoint lap {self.completed_laps}. Continuing."
                )

            self.path_cells = []
            self.path_world = []
            self.last_goal_distance = None
            self.last_progress_time = now
            self.stuck_recovery_count = 0
            return

        # -------------------------------------------------
        # Progress check
        # -------------------------------------------------
        if self.last_goal_distance is None:
            self.last_goal_distance = distance_to_goal
            self.last_progress_time = now

        progress = self.last_goal_distance - distance_to_goal

        if progress > 0.04:
            self.last_goal_distance = distance_to_goal
            self.last_progress_time = now

            if self.stuck_recovery_count > 0:
                self.stuck_recovery_count -= 1

        if now - self.last_progress_time > 5.0:
            self.stuck_recovery_count += 1

            self.get_logger().warn(
                f"Low progress. Recovery attempt "
                f"{self.stuck_recovery_count}/3."
            )

            self.path_cells = []
            self.path_world = []

            self.recovery_direction = self.choose_escape_direction(
                dist_left,
                dist_right,
                obstacle_angle=guard_angle,
                side_left_min=side_left_min,
                side_right_min=side_right_min,
            )

            self.recovery_until = now + 0.75
            self.last_goal_distance = distance_to_goal
            self.last_progress_time = now

            if self.stuck_recovery_count >= 3:
                self.get_logger().warn(
                    "Still stuck after recovery attempts. Skipping waypoint."
                )

                self.current_waypoint = (self.current_waypoint + 1) % len(self.waypoints)
                self.path_cells = []
                self.path_world = []
                self.stuck_recovery_count = 0
                self.last_goal_distance = None
                self.last_progress_time = now

        if now < self.recovery_until:
            self.state = ExplorerState.RECOVERY_TURN
            self.publish_cmd(
                0.0,
                self.recovery_direction * 0.85,
                smooth=False,
            )
            return

        # -------------------------------------------------
        # A* replanning
        # -------------------------------------------------
        should_replan = (
            not self.path_world
            or now - self.last_plan_time > self.replan_period
        )

        if should_replan:
            planned = self.plan_path_to_waypoint(robot_x, robot_y)
            self.last_plan_time = now
        else:
            planned = bool(self.path_world)

        # -------------------------------------------------
        # Map path following
        # -------------------------------------------------
        if planned and self.path_world:
            self.state = ExplorerState.MAP_PLAN

            caution = self.is_caution(sectors)
            lookahead = self.lookahead_caution if caution else self.lookahead_open

            target = self.choose_lookahead_point(
                robot_x,
                robot_y,
                lookahead,
            )

            if target is not None:
                tx, ty = target

                dx = tx - robot_x
                dy = ty - robot_y

                target_angle = math.atan2(dy, dx)
                heading_error = wrap_angle(target_angle - robot_yaw)

                angular = 1.15 * heading_error
                angular += self.wall_clearance_bias(sectors)
                angular = max(-1.00, min(1.00, angular))

                linear = self.speed_from_clearance(
                    sectors,
                    heading_error,
                )

                self.get_logger().info(
                    f"MAP_PLAN: wp={self.current_waypoint + 1}/12, "
                    f"path={len(self.path_world)}, "
                    f"target=({tx:.2f},{ty:.2f}), "
                    f"center={dist_center:.2f}, "
                    f"guard={guard_min:.2f}@{guard_angle:.2f}, "
                    f"wide={wide_min:.2f}@{wide_angle:.2f}, "
                    f"SL={side_left_min:.2f}, SR={side_right_min:.2f}, "
                    f"heading={heading_error:.2f}, "
                    f"v={linear:.2f}, w={angular:.2f}",
                    throttle_duration_sec=1.0,
                )

                self.publish_cmd(linear, angular)
                return

        # -------------------------------------------------
        # Local fallback
        # -------------------------------------------------
        self.state = ExplorerState.LOCAL_FALLBACK

        goal_angle_local = math.atan2(local_dy, local_dx)
        heading_error_local = wrap_angle(goal_angle_local - self.current_yaw)

        if dist_center > 0.80 and guard_min > 0.45:
            local_angle = heading_error_local
        else:
            local_angle = self.fallback_local_angle(
                sectors,
                heading_error_local,
            )

        if dist_center > 1.00:
            local_angle = max(-0.55, min(0.55, local_angle))
        elif dist_center > 0.70:
            local_angle = max(-0.75, min(0.75, local_angle))

        angular = 1.05 * local_angle
        angular += self.wall_clearance_bias(sectors)
        angular = max(-1.00, min(1.00, angular))

        linear = self.speed_from_clearance(
            sectors,
            local_angle,
        )

        self.get_logger().info(
            f"LOCAL_FALLBACK: wp={self.current_waypoint + 1}/12, "
            f"center={dist_center:.2f}, "
            f"guard={guard_min:.2f}@{guard_angle:.2f}, "
            f"wide={wide_min:.2f}@{wide_angle:.2f}, "
            f"L={dist_left:.2f}, R={dist_right:.2f}, "
            f"SL={side_left_min:.2f}, SR={side_right_min:.2f}, "
            f"a={local_angle:.2f}, "
            f"v={linear:.2f}, w={angular:.2f}",
            throttle_duration_sec=1.0,
        )

        self.publish_cmd(linear, angular)


def main(args=None):
    rclpy.init(
        args=args,
        signal_handler_options=SignalHandlerOptions.NO,
    )

    node = SlamExplorer()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt received.")

    finally:
        if not node.shutdown_requested:
            node.stop_robot()

        node.save_debug_outputs()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()