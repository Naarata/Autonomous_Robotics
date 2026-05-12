#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions

from geometry_msgs.msg import TwistStamped, Quaternion
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from nav2_msgs.srv import SaveMap

import os
import statistics
import math


# IMPORTANT: Change this to your actual team package name
TEAM_PACKAGE_NAME = 'ele434_team01_2026'

# --- Timing Constants ---
MAP_SAVE_TIME = 85.0
SHUTDOWN_TIME = 90.0

# --- Safety Distances ---
SAFE_DIST = 0.38
DANGER_DIST = 0.20

# --- Speed Constants ---
MAX_SPEED = 0.15
TURN_SPEED = 0.8
REVERSE_SPEED = 0.06

# --- Waypoint Constants ---
WAYPOINT_TOLERANCE = 0.35       # Robot counts waypoint as visited within 35 cm
WAYPOINT_ANGULAR_GAIN = 1.2     # How strongly it turns towards waypoint
MAX_WAYPOINT_TURN = 0.8         # Limit angular speed while waypoint-following
HEADING_SLOWDOWN_ANGLE = 0.6    # If heading error is large, reduce forward speed

# --- Anti-Stuck Constants ---
MAX_CONSECUTIVE_TURNS = 40
RECOVERY_STEPS = 20


def quaternion_to_euler(orientation: Quaternion):
    """
    Converts quaternion orientation from odometry into roll, pitch, yaw.
    Only yaw is needed for 2D robot navigation.
    """
    x = orientation.x
    y = orientation.y
    z = orientation.z
    w = orientation.w

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = max(min(t2, +1.0), -1.0)
    pitch = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    return roll, pitch, yaw


def clamp(value, min_value, max_value):
    """Limits value between min_value and max_value."""
    return max(min(value, max_value), min_value)


def normalise_angle(angle):
    """Normalises angle to the range -pi to +pi."""
    return math.atan2(math.sin(angle), math.cos(angle))


class ExplorerNode(Node):
    def __init__(self):
        super().__init__('explorer_node')

        # --- Publishers & Subscribers ---
        self.cmd_vel_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile_sensor_data
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        # --- Service Client for SLAM Map Saving ---
        self.map_saver_client = self.create_client(SaveMap, '/map_saver/save_map')

        # --- Timers & State ---
        self.timer = self.create_timer(0.1, self.control_loop)
        self.start_time = None
        self.map_saved = False
        self.is_shutdown = False

        # --- Sensor Readings ---
        self.front_dist = 3.5
        self.left_dist = 3.5
        self.right_dist = 3.5

        # --- Odometry State ---
        self.odom_received = False
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0

        # --- Waypoints ---
        # These are approximate map zones for the robot to visit.
        # Adjust values if your arena is smaller/larger.
        self.waypoints = [
            (1.5, 1.5),
            (1.5, 0.5),
            (1.5, -0.5),
            (1.5, -1.5),
            (0.5, -1.5),
            (-0.5, -1.5),
            (-1.5, -1.5),
            (-1.5, -0.5),
            (-1.5, 0.5),
            (-1.5, 1.5),
            (-0.5, 1.5),
            (0.5, 1.5),
        ]

        self.current_waypoint = 0
        self.visited_waypoints = set()

        # --- FSM State ---
        self.state = 'EXPLORE'

        # --- Anti-Stuck Tracking ---
        self.consecutive_turns = 0
        self.recovery_steps_left = 0
        self.last_turn_direction = 1

        self.get_logger().info("Explorer Node with waypoint memory started.")

    # -------------------------------------------------------------------------
    # SENSOR PROCESSING
    # -------------------------------------------------------------------------

    def get_robust_distance(self, ranges_slice):
        """
        Returns a robust minimum distance from a slice of LiDAR ranges.
        Uses median of closest valid points to reduce noise sensitivity.
        """
        valid_points = [r for r in ranges_slice if 0.1 < r < 3.5]

        if not valid_points:
            return 3.5

        valid_points.sort()
        closest = valid_points[:min(5, len(valid_points))]

        return statistics.median(closest)

    def scan_callback(self, msg):
        """Processes LiDAR data into front, left, and right distance zones."""
        ranges = list(msg.ranges)
        n = len(ranges)

        if n == 0:
            return

        step = max(1, n // 360)

        front_indices = list(range(max(0, n - 20 * step), n)) + list(range(0, min(n, 20 * step)))
        left_indices = list(range(min(n, 70 * step), min(n, 110 * step)))
        right_indices = list(range(min(n, 250 * step), min(n, 290 * step)))

        self.front_dist = self.get_robust_distance([ranges[i] for i in front_indices])
        self.left_dist = self.get_robust_distance([ranges[i] for i in left_indices])
        self.right_dist = self.get_robust_distance([ranges[i] for i in right_indices])

    def odom_callback(self, msg):
        """Stores robot position and yaw from odometry."""
        pose = msg.pose.pose
        _, _, yaw = quaternion_to_euler(pose.orientation)

        self.current_x = pose.position.x
        self.current_y = pose.position.y
        self.current_yaw = yaw
        self.odom_received = True

    # -------------------------------------------------------------------------
    # COMMAND AND MAP SAVING
    # -------------------------------------------------------------------------

    def _make_cmd(self, linear_x=0.0, angular_z=0.0):
        """Builds a TwistStamped message."""
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = float(linear_x)
        msg.twist.angular.z = float(angular_z)
        return msg

    def save_map(self):
        """Calls the map_saver service asynchronously to save the SLAM map."""
        if not self.map_saver_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error('Map saver service unavailable! Map will NOT be saved.')
            return

        home_dir = os.path.expanduser('~')
        map_folder = os.path.join(home_dir, 'ros2_ws', 'src', TEAM_PACKAGE_NAME, 'maps')
        os.makedirs(map_folder, exist_ok=True)

        map_path = os.path.join(map_folder, 'explore_map')

        self.get_logger().info(f'Saving map to: {map_path}')

        request = SaveMap.Request()
        request.map_topic = '/map'
        request.map_url = map_path
        request.image_format = 'png'
        request.map_mode = 'trinary'
        request.free_thresh = 0.25
        request.occupied_thresh = 0.65

        future = self.map_saver_client.call_async(request)
        future.add_done_callback(self.map_save_callback)

    def map_save_callback(self, future):
        try:
            response = future.result()
            if response.result:
                self.get_logger().info('SUCCESS: Map saved!')
            else:
                self.get_logger().error('Map saver returned FAILURE.')
        except Exception as e:
            self.get_logger().error(f'Map save service call failed: {e}')

    # -------------------------------------------------------------------------
    # WAYPOINT LOGIC
    # -------------------------------------------------------------------------

    def get_current_goal(self):
        """
        Returns the current waypoint.
        If all waypoints are visited, restart the loop so the robot keeps moving
        until the 90 s timer ends.
        """
        if self.current_waypoint >= len(self.waypoints):
            self.get_logger().info('All waypoints visited once. Restarting waypoint loop.')
            self.current_waypoint = 0
            self.visited_waypoints.clear()

        return self.waypoints[self.current_waypoint]

    def waypoint_command(self):
        """
        Drives towards the current waypoint using odometry.
        Obstacle avoidance still has priority in the FSM.
        """
        if not self.odom_received:
            # If odometry is not ready, fall back to simple forward exploration.
            return self._make_cmd(linear_x=MAX_SPEED)

        goal_x, goal_y = self.get_current_goal()

        error_x = goal_x - self.current_x
        error_y = goal_y - self.current_y
        distance_to_goal = math.hypot(error_x, error_y)

        if distance_to_goal < WAYPOINT_TOLERANCE:
            self.visited_waypoints.add(self.current_waypoint)
            self.get_logger().info(
                f'Visited waypoint {self.current_waypoint + 1}/{len(self.waypoints)} '
                f'at ({goal_x:.1f}, {goal_y:.1f})'
            )
            self.current_waypoint += 1
            return self._make_cmd()

        desired_yaw = math.atan2(error_y, error_x)
        yaw_error = normalise_angle(desired_yaw - self.current_yaw)

        angular_z = clamp(
            WAYPOINT_ANGULAR_GAIN * yaw_error,
            -MAX_WAYPOINT_TURN,
            MAX_WAYPOINT_TURN
        )

        # If the robot is facing roughly towards the waypoint, move forward.
        # If not, turn more and move slowly to avoid large arcs.
        if abs(yaw_error) < HEADING_SLOWDOWN_ANGLE:
            linear_x = MAX_SPEED
        else:
            linear_x = MAX_SPEED * 0.35

        return self._make_cmd(linear_x=linear_x, angular_z=angular_z)

    # -------------------------------------------------------------------------
    # MAIN CONTROL LOOP
    # -------------------------------------------------------------------------

    def control_loop(self):
        """10 Hz control loop implementing time tracking and navigation FSM."""

        if self.is_shutdown:
            return

        now = self.get_clock().now().nanoseconds / 1e9

        # Wait for LiDAR before starting the 90 s clock
        if self.start_time is None:
            if self.front_dist < 3.5:
                self.start_time = now
                self.get_logger().info('LiDAR active. Clock started!')
            return

        elapsed = now - self.start_time

        # --- TRIGGER MAP SAVE ---
        if elapsed >= MAP_SAVE_TIME and not self.map_saved:
            self.get_logger().info(f'{MAP_SAVE_TIME:.0f}s reached — saving map.')
            self.save_map()
            self.map_saved = True

        # --- HARD SHUTDOWN ---
        if elapsed >= SHUTDOWN_TIME:
            self.get_logger().info('90s reached. Stopping robot.')
            self.cmd_vel_pub.publish(self._make_cmd())
            self.is_shutdown = True
            raise KeyboardInterrupt

        # --- FSM ---
        cmd = self._run_fsm()
        self.cmd_vel_pub.publish(cmd)

    def _run_fsm(self):
        """
        FSM with waypoint-guided exploration:
          EXPLORE  — move towards next waypoint
          AVOID    — turn away from obstacles
          REVERSE  — reverse if too close
          RECOVER  — escape turning loops
        """

        if self.state == 'EXPLORE':
            if self.front_dist < DANGER_DIST:
                self.get_logger().warn(f'DANGER ({self.front_dist:.2f} m) — reversing!')
                self.state = 'REVERSE'

            elif self.front_dist < SAFE_DIST:
                self.get_logger().info(f'Obstacle at {self.front_dist:.2f} m — avoiding.')
                self.consecutive_turns = 0
                self.state = 'AVOID'

            else:
                self.consecutive_turns = 0
                return self.waypoint_command()

        if self.state == 'AVOID':
            if self.front_dist < DANGER_DIST:
                self.state = 'REVERSE'

            elif self.front_dist > (SAFE_DIST + 0.05):
                self.state = 'EXPLORE'

            elif self.consecutive_turns >= MAX_CONSECUTIVE_TURNS:
                self.get_logger().warn('Stuck turning! Switching to RECOVERY.')
                self.recovery_steps_left = RECOVERY_STEPS
                self.state = 'RECOVER'

            else:
                gap = self.left_dist - self.right_dist

                if abs(gap) > 0.1:
                    self.last_turn_direction = 1 if gap > 0 else -1

                self.consecutive_turns += 1

                return self._make_cmd(
                    angular_z=self.last_turn_direction * TURN_SPEED
                )

        if self.state == 'REVERSE':
            if self.front_dist > SAFE_DIST:
                self.state = 'AVOID'
            else:
                return self._make_cmd(linear_x=-REVERSE_SPEED)

        if self.state == 'RECOVER':
            if self.recovery_steps_left > 0:
                self.recovery_steps_left -= 1

                return self._make_cmd(
                    linear_x=MAX_SPEED * 0.5,
                    angular_z=self.last_turn_direction * TURN_SPEED * 0.5
                )

            else:
                self.get_logger().info('Recovery complete — resuming EXPLORE.')
                self.consecutive_turns = 0
                self.state = 'EXPLORE'

        return self._make_cmd()


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------

def main(args=None):
    rclpy.init(
        args=args,
        signal_handler_options=SignalHandlerOptions.NO
    )

    node = ExplorerNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info('Shutting down — killing motors.')
        node.cmd_vel_pub.publish(node._make_cmd())

    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

