#!/usr/bin/env python3
# killgz
# ros2 launch tuos_task_sims obstacle_avoidance.launch.py with_gui:=false
# ros2 launch tuos_tb3_tools slam.launch.py use_sim_time:=false


import rclpy
from rclpy.node import Node


from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TwistStamped, Quaternion
from nav_msgs.msg import OccupancyGrid


import math
import heapq
import numpy as np
from scipy.ndimage import distance_transform_edt


# --- THE FIX: Import TF2 for real-world localization ---
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException


def quaternion_to_euler(orientation):
    x, y, z, w = orientation.x, orientation.y, orientation.z, orientation.w
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return 0.0, 0.0, yaw


class RealWorldAStarNavigator(Node):
    def __init__(self):
        super().__init__("astar_navigator")


        # REAL-WORLD TUNING PARAMETERS
        self.HARD_INFLATION_M = 0.22
        self.SOFT_ZONE_M = 0.45
        self.SOFT_COST_WEIGHT = 2.0
        self.STEP_REACH_DIST = 0.10
        self.WAYPOINT_REACH_DIST = 0.25
        self.DANGER_DIST = 0.25
        self.DANGER_ARC = math.radians(30)


        self.vel_pub = self.create_publisher(TwistStamped, "/cmd_vel", 10)
       
        # --- THE FIX: Replace /odom with the TF Buffer ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)


        self.map_sub = self.create_subscription(OccupancyGrid, "/map", self.map_callback, 10)
        self.laser_sub = self.create_subscription(LaserScan, "/scan", self.laser_callback, 10)


        # Timer for control loop (20Hz)
        self.timer = self.create_timer(0.05, self.control_loop)


        # Robot State
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.odom_ready = False # Will flip to true once TF connects


        # LiDAR & Map State
        self.min_front_dist = float('inf')
        self.map_data = None
        self.map_width = 0
        self.map_height = 0
        self.map_res = 0.0
        self.map_origin_x = 0.0
        self.map_origin_y = 0.0
        self.map_ready = False
        self.dist_map = None
        self.cost_map_ready = False


        self.waypoints = [
            (1.4,   1.4), (1.25,  0.25), (1.25, -0.25),
            (1.4,  -1.4), (0.25, -1.25), (-0.5, -1.25),
            (-1.4, -1.4), (-1.25, -0.25), (-1.25,  0.25),
            (-1.4,  1.4), (-0.25, 1.25), (0.4,  1.4),
        ]
        self.current_wp_idx = 0
        self.path = []        
        self.state = 'WAITING_FOR_DATA'  


        self.get_logger().info("🚀 TF2 A* Tracker Initialized!")


    def map_callback(self, msg: OccupancyGrid):
        self.map_res = msg.info.resolution
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_origin_x = msg.info.origin.position.x
        self.map_origin_y = msg.info.origin.position.y
        self.map_data = msg.data
        self.map_ready = True
       
        grid = np.array(self.map_data, dtype=np.int8).reshape(self.map_height, self.map_width)
        obstacle_mask = (grid > 50) | (grid == -1)
        self.dist_map = distance_transform_edt(~obstacle_mask)
        self.cost_map_ready = True


    def laser_callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges)
        angles = np.arange(len(ranges)) * msg.angle_increment + msg.angle_min
        mask = np.abs(angles) <= self.DANGER_ARC
        front_ranges = ranges[mask]
        valid = front_ranges[(front_ranges >= msg.range_min) & (front_ranges <= msg.range_max)]
       
        if len(valid) >= 3:
            sorted_valid = np.sort(valid)
            self.min_front_dist = float(np.mean(sorted_valid[:3]))
        else:
            self.min_front_dist = float('inf')


    def get_hard_inflation_cells(self):
        if self.map_res == 0.0: return 4
        return self.HARD_INFLATION_M / self.map_res


    def get_soft_inflation_cells(self):
        if self.map_res == 0.0: return 9
        return self.SOFT_ZONE_M / self.map_res


    def world_to_grid(self, x, y):
        gx = int(math.floor((x - self.map_origin_x) / self.map_res))
        gy = int(math.floor((y - self.map_origin_y) / self.map_res))
        return gx, gy


    def grid_to_world(self, gx, gy):
        wx = (gx * self.map_res) + self.map_origin_x + (self.map_res / 2.0)
        wy = (gy * self.map_res) + self.map_origin_y + (self.map_res / 2.0)
        return wx, wy


    def is_blocked(self, gx, gy):
        if gx < 0 or gx >= self.map_width or gy < 0 or gy >= self.map_height:
            return True
        if not self.cost_map_ready:
            idx = (gy * self.map_width) + gx
            return self.map_data[idx] > 50 or self.map_data[idx] == -1
        return self.dist_map[gy, gx] < self.get_hard_inflation_cells()


    def get_cell_cost(self, gx, gy):
        if not self.cost_map_ready: return 0.0
        d = self.dist_map[gy, gx]
        soft_cells = self.get_soft_inflation_cells()
        if d >= soft_cells: return 0.0
        ratio = 1.0 - (d / soft_cells)
        return self.SOFT_COST_WEIGHT * (ratio ** 2)


    def line_of_sight(self, world_a, world_b):
        x0, y0 = self.world_to_grid(*world_a)
        x1, y1 = self.world_to_grid(*world_b)
        dx = abs(x1 - x0); dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1; sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            if self.is_blocked(x0, y0): return False
            if x0 == x1 and y0 == y1: return True
            e2 = 2 * err
            if e2 > -dy: err -= dy; x0 += sx
            if e2 < dx: err += dx; y0 += sy


    def smooth_path(self, path):
        if len(path) < 3: return path
        smoothed = [path[0]]; i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if self.line_of_sight(path[i], path[j]): break
                j -= 1
            smoothed.append(path[j])
            i = j
        return smoothed


    def a_star_plan(self, start_world, goal_world):
        start_g = self.world_to_grid(*start_world)
        goal_g = self.world_to_grid(*goal_world)


        if self.is_blocked(*start_g):
            self.get_logger().warn("⚠️ START cell trapped! Recovering...")
            found_clear_start = False
            for r in range(1, 15):
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        if max(abs(dx), abs(dy)) == r:
                            test_g = (start_g[0] + dx, start_g[1] + dy)
                            if not self.is_blocked(*test_g):
                                start_g = test_g
                                found_clear_start = True
                                break
                    if found_clear_start: break
                if found_clear_start: break
            if not found_clear_start: return []


        if self.is_blocked(*goal_g):
            self.get_logger().warn("⚠️ GOAL cell blocked! Recovering...")
            found_clear_goal = False
            for r in range(1, 15):
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        if max(abs(dx), abs(dy)) == r:
                            test_g = (goal_g[0] + dx, goal_g[1] + dy)
                            if not self.is_blocked(*test_g):
                                goal_g = test_g
                                found_clear_goal = True
                                break
                    if found_clear_goal: break
                if found_clear_goal: break
            if not found_clear_goal: return []


        open_set = []
        heapq.heappush(open_set, (0, start_g))
        came_from = {}
        g_score = {start_g: 0}


        def heuristic(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])


        while open_set:
            _, current = heapq.heappop(open_set)


            if current == goal_g:
                path = []
                while current in came_from:
                    path.append(self.grid_to_world(*current))
                    current = came_from[current]
                path.reverse()
                return path


            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if self.is_blocked(*neighbor): continue


                move_cost = 1.414 if dx != 0 and dy != 0 else 1.0
                move_cost += self.get_cell_cost(*neighbor)
                tentative_g = g_score[current] + move_cost


                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal_g)
                    heapq.heappush(open_set, (f_score, neighbor))
        return []


    def control_loop(self):
        msg = TwistStamped()


        # --- THE FIX: Look up the robot's physical position on the SLAM map ---
        try:
            # We ask TF: "Where is the robot's footprint (base_footprint) relative to the SLAM map (map)?"
            t = self.tf_buffer.lookup_transform('map', 'base_footprint', rclpy.time.Time())
            self.current_x = t.transform.translation.x
            self.current_y = t.transform.translation.y
            _, _, yaw = quaternion_to_euler(t.transform.rotation)
            self.current_yaw = yaw
            self.odom_ready = True
        except (LookupException, ConnectivityException, ExtrapolationException):
            self.get_logger().info('Waiting for SLAM to align with robot position...', once=True)
            return


        if not self.map_ready or not self.odom_ready: return


        if self.current_wp_idx >= len(self.waypoints):
            self.get_logger().info("🏆 All Waypoints Reached!", once=True)
            self.vel_pub.publish(msg)
            return


        target_x, target_y = self.waypoints[self.current_wp_idx]


        if self.state == 'WAITING_FOR_DATA':
            self.state = 'PLANNING'


        elif self.state == 'PLANNING':
            raw_path = self.a_star_plan((self.current_x, self.current_y), (target_x, target_y))


            if raw_path:
                self.path = self.smooth_path(raw_path)
                self.state = 'DRIVING'
            else:
                self.get_logger().error(f"❌ Retrying path to WP {self.current_wp_idx}...")
                self.vel_pub.publish(msg)
                return


        elif self.state == 'DRIVING':
            if self.min_front_dist < self.DANGER_DIST:
                msg.twist.linear.x = 0.0; msg.twist.angular.z = -0.6
                self.vel_pub.publish(msg); return


            dist_to_goal = math.hypot(target_x - self.current_x, target_y - self.current_y)
            if dist_to_goal < self.WAYPOINT_REACH_DIST:
                self.get_logger().info(f"🎯 Waypoint {self.current_wp_idx} Cleared!")
                self.current_wp_idx += 1
                self.state = 'PLANNING'
                self.vel_pub.publish(msg)
                return


            if len(self.path) == 0:
                self.state = 'PLANNING'; return


            next_step_x, next_step_y = self.path[0]
            if math.hypot(next_step_x - self.current_x, next_step_y - self.current_y) < self.STEP_REACH_DIST:
                self.path.pop(0)
                if len(self.path) == 0: return
                next_step_x, next_step_y = self.path[0]


            angle_to_step = math.atan2(next_step_y - self.current_y, next_step_x - self.current_x)
            yaw_error = math.atan2(math.sin(angle_to_step - self.current_yaw), math.cos(angle_to_step - self.current_yaw))


            msg.twist.angular.z = 4.0 * yaw_error
            if abs(yaw_error) < 0.20:
                msg.twist.linear.x = 0.18 * math.cos(yaw_error)
            else:
                msg.twist.linear.x = 0.0
               
            self.vel_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = RealWorldAStarNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.vel_pub.publish(TwistStamped())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

