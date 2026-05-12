#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import LaserScan
from nav2_msgs.srv import SaveMap
import os
import statistics

# IMPORTANT: Change this to your actual team package name!
TEAM_PACKAGE_NAME = 'ele434_team01_2026'

# --- Timing Constants ---
MAP_SAVE_TIME = 85.0    # Save map at 85s (extra buffer before shutdown)
SHUTDOWN_TIME = 90.0    # Hard stop at 90s

# --- Safety Distances ---
SAFE_DIST   = 0.38      # Start avoiding at 38cm
DANGER_DIST = 0.20      # Emergency reverse if closer than 20cm

# --- Speed Constants ---
MAX_SPEED   = 0.15      # Linear speed (m/s)
TURN_SPEED  = 0.8       # Rotational speed (rad/s)
REVERSE_SPEED = 0.06    # Reverse speed (m/s)

# --- Anti-Stuck Constants ---
MAX_CONSECUTIVE_TURNS = 40   # ~4 seconds of turning before trying recovery
RECOVERY_STEPS        = 20   # ~2 seconds of recovery behaviour


class ExplorerNode(Node):
    def __init__(self):
        super().__init__('explorer_node')

        # --- Publishers & Subscribers ---
        # TwistStamped is required by this robot's controller (not plain Twist)
        self.cmd_vel_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)

        # Leading slash = absolute topic (avoids namespace issues)
        # SensorDataQoS matches the LiDAR driver's QoS profile
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile_sensor_data
        )

        # --- Service Client for SLAM Map Saving ---
        self.map_saver_client = self.create_client(SaveMap, '/map_saver/save_map')

        # --- Timers & State ---
        self.timer = self.create_timer(0.1, self.control_loop)  # 10 Hz
        self.start_time   = None
        self.map_saved    = False
        self.is_shutdown  = False

        # --- Sensor Readings (default to "clear" until LiDAR boots) ---
        self.front_dist = 3.5
        self.left_dist  = 3.5
        self.right_dist = 3.5

        # --- FSM State ---
        self.state = 'EXPLORE'

        # --- Anti-Stuck Tracking ---
        self.consecutive_turns    = 0
        self.recovery_steps_left  = 0
        self.last_turn_direction  = 1   # 1 = left, -1 = right (memory to avoid flip-flopping)

        self.get_logger().info("Explorer Node started. Waiting for first LiDAR scan...")

    # -------------------------------------------------------------------------
    # SENSOR PROCESSING
    # -------------------------------------------------------------------------

    def get_robust_distance(self, ranges_slice):
        """
        Returns a robust minimum distance from a slice of LiDAR ranges.

        Improvement over original:
          - Uses MEDIAN of the 5 closest valid points instead of MEAN of 3.
          - Median is more resistant to outlier noise spikes common on real hardware.
        """
        valid_points = [r for r in ranges_slice if 0.1 < r < 3.5]

        if not valid_points:
            return 3.5  # Assume clear if no valid data

        valid_points.sort()
        closest = valid_points[:min(5, len(valid_points))]

        # Median is more robust to single noisy spikes than mean
        return statistics.median(closest)

    def scan_callback(self, msg):
        """Processes LiDAR data into front, left, and right distance zones."""
        ranges = list(msg.ranges)
        n = len(ranges)

        if n == 0:
            return

        # Use proportional slicing so code works with different LiDAR resolutions
        # (e.g., 360-point or 720-point scans)
        step = n // 360

        front_indices = list(range(n - 20 * step, n)) + list(range(0, 20 * step))
        left_indices  = list(range(70 * step, 110 * step))
        right_indices = list(range(250 * step, 290 * step))

        self.front_dist = self.get_robust_distance([ranges[i] for i in front_indices])
        self.left_dist  = self.get_robust_distance([ranges[i] for i in left_indices])
        self.right_dist = self.get_robust_distance([ranges[i] for i in right_indices])

    def _make_cmd(self, linear_x=0.0, angular_z=0.0):
        """Build a TwistStamped message (required by this robot's controller)."""
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

        home_dir   = os.path.expanduser('~')
        map_folder = os.path.join(home_dir, 'ros2_ws', 'src', TEAM_PACKAGE_NAME, 'maps')
        os.makedirs(map_folder, exist_ok=True)
        map_path   = os.path.join(map_folder, 'explore_map')

        self.get_logger().info(f'Saving map to: {map_path}')

        request = SaveMap.Request()
        request.map_topic       = '/map'
        request.map_url         = map_path
        request.image_format    = 'png'
        request.map_mode        = 'trinary'
        request.free_thresh     = 0.25
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
    # MAIN CONTROL LOOP (FSM)
    # -------------------------------------------------------------------------

    def control_loop(self):
        """10 Hz control loop implementing the navigation FSM and time tracking."""

        if self.is_shutdown:
            return

        now = self.get_clock().now().nanoseconds / 1e9

        # Wait for LiDAR to boot before starting the clock
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
            # Note: robot continues navigating after save (unlike original)

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
        Finite State Machine with 4 states:
          EXPLORE  — drive forward
          AVOID    — turn in place toward open space
          REVERSE  — back up when too close
          RECOVER  — escape getting stuck in a turning loop
        """

        if self.state == 'EXPLORE':
            if self.front_dist < DANGER_DIST:
                self.get_logger().warn(f'DANGER ({self.front_dist:.2f}m) — reversing!')
                self.state = 'REVERSE'
            elif self.front_dist < SAFE_DIST:
                self.get_logger().info(f'Obstacle at {self.front_dist:.2f}m — avoiding.')
                self.consecutive_turns = 0
                self.state = 'AVOID'
            else:
                self.consecutive_turns = 0
                return self._make_cmd(linear_x=MAX_SPEED)

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
                return self._make_cmd(angular_z=self.last_turn_direction * TURN_SPEED)

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

        return self._make_cmd()  # Default: stop


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------

def main(args=None):
    rclpy.init(
        args=args,
        signal_handler_options=SignalHandlerOptions.NO  # Matches friend's working config
    )
    node = ExplorerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down — killing motors.')
        node.cmd_vel_pub.publish(node._make_cmd())  # Stop motors

    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()

