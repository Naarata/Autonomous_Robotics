#!/usr/bin/env python3

import math
import time
from enum import Enum

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TwistStamped, Quaternion
from nav_msgs.msg import Odometry


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def quaternion_to_yaw(orientation: Quaternion) -> float:
    """Convert quaternion to yaw angle in radians."""
    x = orientation.x
    y = orientation.y
    z = orientation.z
    w = orientation.w

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)

    return math.atan2(t3, t4)


class ExplorerState(Enum):
    GO_TO_WAYPOINT = 1
    GAP_AVOID = 2
    EMERGENCY_ESCAPE = 3
    RECOVERY_TURN = 4


class SlamExplorer(Node):

    def __init__(self):
        super().__init__("slam_explorer")

        # -------------------------------------------------
        # Parameters
        # -------------------------------------------------
        self.declare_parameter("run_duration", 90.0)
        self.declare_parameter("max_speed", 0.17)
        self.declare_parameter("cruise_speed", 0.15)
        self.declare_parameter("corner_speed", 0.09)
        self.declare_parameter("max_angular_speed", 0.95)
        self.declare_parameter("goal_tolerance", 0.23)

        self.run_duration = float(self.get_parameter("run_duration").value)
        self.max_speed = float(self.get_parameter("max_speed").value)
        self.cruise_speed = float(self.get_parameter("cruise_speed").value)
        self.corner_speed = float(self.get_parameter("corner_speed").value)
        self.max_angular_speed = float(self.get_parameter("max_angular_speed").value)
        self.goal_tolerance = float(self.get_parameter("goal_tolerance").value)

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

        self.vel_pub = self.create_publisher(
            TwistStamped,
            "/cmd_vel",
            10,
        )

        self.timer = self.create_timer(
            1.0 / 20.0,
            self.navigation_control,
        )

        # -------------------------------------------------
        # Robot state
        # -------------------------------------------------
        self.scan_data = None
        self.have_odom = False
        self.shutdown_requested = False

        self.initialised_odom = False
        self.x_offset = 0.0
        self.y_offset = 0.0
        self.yaw_offset = 0.0

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0

        self.start_time = None

        # -------------------------------------------------
        # Outer-zone waypoints for 4m x 4m arena
        # -------------------------------------------------
        # Slightly inside the outer wall.
        self.waypoints = [
            (1.45, 1.45),
            (1.45, 0.50),
            (1.45, -0.50),
            (1.45, -1.45),
            (0.50, -1.45),
            (-0.50, -1.45),
            (-1.45, -1.45),
            (-1.45, -0.50),
            (-1.45, 0.50),
            (-1.45, 1.45),
            (-0.50, 1.45),
            (0.50, 1.45),
        ]

        self.current_waypoint = 0
        self.completed_laps = 0

        # -------------------------------------------------
        # Avoidance / recovery state
        # -------------------------------------------------
        self.state = ExplorerState.GO_TO_WAYPOINT

        self.recovery_until = 0.0
        self.recovery_direction = 1.0

        self.last_goal_distance = None
        self.last_progress_time = time.time()

        self.get_logger().info("slam_explorer node initialised.")

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

        # Rotate odometry into the robot's initial heading frame.
        # This only affects our waypoint logic; it does NOT publish any TF.
        c = math.cos(-self.yaw_offset)
        s = math.sin(-self.yaw_offset)

        self.current_x = c * dx - s * dy
        self.current_y = s * dx + c * dy
        self.current_yaw = wrap_angle(yaw - self.yaw_offset)

        self.have_odom = True

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

        # Navigation sectors
        center_mask = (angles > -0.55) & (angles < 0.55)
        left_mask = (angles >= 0.55) & (angles < 1.45)
        right_mask = (angles <= -0.55) & (angles > -1.45)
        front_mask = (angles > -1.57) & (angles < 1.57)

        # Emergency sector only checks true front.
        # This prevents side walls from constantly triggering emergency mode.
        emergency_mask = (angles > -0.40) & (angles < 0.40)

        dist_center = robust_distance(center_mask)
        dist_left = robust_distance(left_mask)
        dist_right = robust_distance(right_mask)

        front_values = ranges[front_mask]
        front_angles = angles[front_mask]

        if front_values.size > 0:
            min_index = int(np.argmin(front_values))
            min_dist_front = float(front_values[min_index])
            min_angle_front = float(front_angles[min_index])
        else:
            min_dist_front = safe_max
            min_angle_front = 0.0

        emergency_values = ranges[emergency_mask]
        emergency_angles = angles[emergency_mask]

        if emergency_values.size > 0:
            emergency_index = int(np.argmin(emergency_values))
            emergency_dist = float(emergency_values[emergency_index])
            emergency_angle = float(emergency_angles[emergency_index])
        else:
            emergency_dist = safe_max
            emergency_angle = 0.0

        return {
            "center": dist_center,
            "left": dist_left,
            "right": dist_right,
            "front_min": min_dist_front,
            "front_min_angle": min_angle_front,
            "emergency_dist": emergency_dist,
            "emergency_angle": emergency_angle,
            "ranges": ranges,
            "angles": angles,
        }

    # -------------------------------------------------
    # Gap-following helpers
    # -------------------------------------------------
    def find_best_gap_angle(self, ranges, angles, goal_angle_local):
        """
        Find a smooth escape direction from the front 180-degree LiDAR view.
        """

        front_mask = (angles > -1.45) & (angles < 1.45)
        front_ranges = ranges[front_mask]
        front_angles = angles[front_mask]

        if front_ranges.size == 0:
            return 0.0

        # Direction is considered passable if there is enough clearance.
        passable = front_ranges > 0.58

        # If waypoint direction is safe, keep bias toward it.
        goal_idx = int(np.argmin(np.abs(front_angles - goal_angle_local)))

        if 0 <= goal_idx < len(passable) and passable[goal_idx]:
            return float(front_angles[goal_idx])

        best_start = None
        best_end = None
        current_start = None

        for i, is_open in enumerate(passable):
            if is_open and current_start is None:
                current_start = i

            if (not is_open or i == len(passable) - 1) and current_start is not None:
                current_end = i if is_open else i - 1

                if best_start is None:
                    best_start = current_start
                    best_end = current_end
                else:
                    current_width = current_end - current_start
                    best_width = best_end - best_start

                    if current_width > best_width:
                        best_start = current_start
                        best_end = current_end

                current_start = None

        # No obvious gap: choose side with greater clearance.
        if best_start is None:
            left_values = front_ranges[front_angles > 0.2]
            right_values = front_ranges[front_angles < -0.2]

            left_clearance = (
                float(np.percentile(left_values, 70))
                if left_values.size > 0
                else 0.0
            )

            right_clearance = (
                float(np.percentile(right_values, 70))
                if right_values.size > 0
                else 0.0
            )

            return 0.95 if left_clearance > right_clearance else -0.95

        gap_center_index = int((best_start + best_end) / 2)
        return float(front_angles[gap_center_index])

    def clearance_speed(self, front_distance):
        """
        Smooth speed control based on forward clearance.
        """
        if front_distance < 0.28:
            return 0.0
        elif front_distance < 0.45:
            return 0.07
        elif front_distance < 0.70:
            return 0.11
        else:
            return self.cruise_speed

    # -------------------------------------------------
    # Motion helpers
    # -------------------------------------------------
    def publish_cmd(self, linear_x: float, angular_z: float):
        linear_x = max(-0.06, min(self.max_speed, linear_x))
        angular_z = max(
            -self.max_angular_speed,
            min(self.max_angular_speed, angular_z),
        )

        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"

        msg.twist.linear.x = float(linear_x)
        msg.twist.angular.z = float(angular_z)

        self.vel_pub.publish(msg)

        # Start 90-second timer when the robot first moves.
        if self.start_time is None and (
            abs(linear_x) > 1e-3 or abs(angular_z) > 1e-3
        ):
            self.start_time = time.time()
            self.get_logger().info("90-second exploration timer started.")

    def stop_robot(self):
        for _ in range(5):
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            msg.twist.linear.x = 0.0
            msg.twist.angular.z = 0.0
            self.vel_pub.publish(msg)
            time.sleep(0.02)

    # -------------------------------------------------
    # Main navigation logic
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
                rclpy.shutdown()
                return

        sectors = self.get_lidar_sectors()
        if sectors is None:
            return

        dist_center = sectors["center"]
        dist_left = sectors["left"]
        dist_right = sectors["right"]

        emergency_dist = sectors["emergency_dist"]
        emergency_angle = sectors["emergency_angle"]

        ranges = sectors["ranges"]
        angles = sectors["angles"]

        # -------------------------------------------------
        # Waypoint tracking
        # -------------------------------------------------
        goal_x, goal_y = self.waypoints[self.current_waypoint]

        dx = goal_x - self.current_x
        dy = goal_y - self.current_y

        distance_to_goal = math.hypot(dx, dy)
        goal_angle_global = math.atan2(dy, dx)
        heading_error = wrap_angle(goal_angle_global - self.current_yaw)

        if distance_to_goal < self.goal_tolerance:
            self.get_logger().info(
                f"Reached outer-zone waypoint {self.current_waypoint + 1}/12"
            )

            self.current_waypoint += 1

            if self.current_waypoint >= len(self.waypoints):
                self.current_waypoint = 0
                self.completed_laps += 1

                self.get_logger().info(
                    f"Completed waypoint lap {self.completed_laps}. Continuing exploration."
                )

            self.last_goal_distance = None
            self.last_progress_time = time.time()
            return

        # -------------------------------------------------
        # Stuck / low-progress detection
        # -------------------------------------------------
        now = time.time()

        if self.last_goal_distance is None:
            self.last_goal_distance = distance_to_goal
            self.last_progress_time = now

        progress = self.last_goal_distance - distance_to_goal

        if progress > 0.04:
            self.last_goal_distance = distance_to_goal
            self.last_progress_time = now

        if now - self.last_progress_time > 3.5:
            self.get_logger().warn(
                "Low progress detected. Skipping waypoint and forcing arc recovery."
            )

            self.current_waypoint = (self.current_waypoint + 1) % len(self.waypoints)

            self.recovery_direction = 1.0 if dist_left > dist_right else -1.0
            self.recovery_until = now + 1.4

            self.last_goal_distance = None
            self.last_progress_time = now

        # -------------------------------------------------
        # 1. Emergency escape
        # -------------------------------------------------
        # Only true front sector triggers this.
        if emergency_dist < 0.20:
            self.state = ExplorerState.EMERGENCY_ESCAPE

            turn = (
                -self.max_angular_speed
                if emergency_angle > 0.0
                else self.max_angular_speed
            )

            self.get_logger().warn(
                f"EMERGENCY_ESCAPE: obstacle {emergency_dist:.2f} m away",
                throttle_duration_sec=0.8,
            )

            self.publish_cmd(-0.025, turn)
            return

        # -------------------------------------------------
        # 2. Timed recovery arc
        # -------------------------------------------------
        if now < self.recovery_until:
            self.state = ExplorerState.RECOVERY_TURN

            self.publish_cmd(
                0.08,
                self.recovery_direction * self.max_angular_speed,
            )
            return

        # -------------------------------------------------
        # 3. Gap-based avoidance
        # -------------------------------------------------
        if dist_center < 0.72:
            self.state = ExplorerState.GAP_AVOID

            gap_angle = self.find_best_gap_angle(
                ranges=ranges,
                angles=angles,
                goal_angle_local=heading_error,
            )

            side_repulsion = 0.0

            if dist_left < 0.34:
                side_repulsion -= 0.45

            if dist_right < 0.34:
                side_repulsion += 0.45

            angular = 1.35 * gap_angle + side_repulsion
            linear = self.clearance_speed(dist_center)

            # In corners, keep moving in an arc instead of dithering.
            if dist_center < 0.48:
                linear = self.corner_speed

            self.publish_cmd(linear, angular)

            self.get_logger().info(
                f"GAP_AVOID: gap={gap_angle:.2f}, "
                f"front={dist_center:.2f}, "
                f"L={dist_left:.2f}, "
                f"R={dist_right:.2f}, "
                f"v={linear:.2f}, "
                f"w={angular:.2f}",
                throttle_duration_sec=1.0,
            )

            return

        # -------------------------------------------------
        # 4. Normal waypoint navigation
        # -------------------------------------------------
        self.state = ExplorerState.GO_TO_WAYPOINT

        angular = 1.45 * heading_error

        if abs(heading_error) > 1.25:
            linear = 0.07
        elif abs(heading_error) > 0.65:
            linear = 0.11
        else:
            linear = self.cruise_speed

        # Light side correction while still moving forward.
        if dist_left < 0.36:
            angular -= 0.35

        if dist_right < 0.36:
            angular += 0.35

        self.publish_cmd(linear, angular)

        self.get_logger().info(
            f"GOAL: wp={self.current_waypoint + 1}/12, "
            f"pos=({self.current_x:.2f},{self.current_y:.2f}), "
            f"dist={distance_to_goal:.2f}, "
            f"heading={heading_error:.2f}, "
            f"front={dist_center:.2f}, "
            f"L={dist_left:.2f}, "
            f"R={dist_right:.2f}",
            throttle_duration_sec=1.2,
        )


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

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()