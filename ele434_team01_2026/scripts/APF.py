#!/usr/bin/env python3


import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions


from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
import time
# from part2_navigation_modules.tb3_tools import quaternion_to_euler
from math import sqrt, pow, pi
import math
import numpy as np
from geometry_msgs.msg import Quaternion
from math import atan2, asin


def quaternion_to_euler(orientation: Quaternion):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    adapted from:
    https://automaticaddison.com/how-to-convert-a-quaternion-into-euler-angles-in-python/
    """
    x = orientation.x
    y = orientation.y
    z = orientation.z
    w = orientation.w
   
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = atan2(t0, t1)
   
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = asin(t2)
   
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = atan2(t3, t4)
   
    return roll, pitch, yaw # in radians

class AutonomousNavigation(Node):


    def __init__(self):
        super().__init__("Navigator")


        self.lidar_sub = self.create_subscription(
            msg_type = LaserScan,
            topic = "/scan",
            callback = self.lidar_callback,
            qos_profile = 10,
        )
        self.vel_pub = self.create_publisher(
            msg_type = TwistStamped,
            topic = "/cmd_vel",
            qos_profile = 10,
        )
        self.odom_data = self.create_subscription(
            msg_type = Odometry,
            topic = "/odom",
            callback = self.odom_callback,
            qos_profile = 10,
        )


        self.timer = self.create_timer(
            timer_period_sec = 1/20,
            callback = self.navigation_control
        )


        # Set current state
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.scan_data = None


        self.waypoints = [
            (1.25, 1.25),
            (1.25, 0.25),
            (1.25, -0.25),
            (1.25, -1.25),
            (0.25, -1.25),
            (-0.5, -1.25),
            (-1.25, -1.25),
            (-1.25, -0.25),
            (-1.25, 0.25),
            (-1.25, 1.25),
            (-0.25, 1.25),
            (0.25, 1.25)
        ]
        # self.waypoints = [
        #     (1.5, -0.5),
        #     (1.5, -1.5),
        #     (0.5, -1.5),
        #     (-0.5, -1.5),
        #     (-1.5, -1.5),
        #     (-1.5, -0.5),
        #     (-1.5, 0.5),
        #     (-1.5, 1.5),
        #     (-0.5, 1.5),
        #     (0.5, 1.5),
        #     (1.5, 1.5),
        #     (1.5, 0.5)
        # ]


        self.current_waypoint = 0
        # STATE MACHINE
        self.state = 'SEEK_GOAL'
        self.distance_to_goal_at_trap = float('inf')
        self.last_escape_time = 0.0
        # Controller
        self.prev_dist_error = 0.0


        # APF Parameters
        self.k_goal = 1.5
        self.k_obstacle = 0.8
        self.distance_safe = 0.4
        self.max_speed = 0.25
        self.max_angular_speed = 1.3
        self.goal_tolerance = 0.3


        self.get_logger().info(f"The '{self.get_name()}' node is initialised.")




    def lidar_callback(self, scan_data:LaserScan):
        self.scan_data = scan_data
   
    def odom_callback(self, msg_data:Odometry):
        pose = msg_data.pose.pose
        (roll, pitch, yaw) = quaternion_to_euler(pose.orientation)
        self.current_x = pose.position.x
        self.current_y = pose.position.y
        self.current_yaw = yaw


    def force_obstacles(self):
        if not self.scan_data:
            return np.array([0.0, 0.0])
       
        force_x = 0.0
        force_y = 0.0
       
        ranges = self.scan_data.ranges
        angle_min = self.scan_data.angle_min
        angle_increment = self.scan_data.angle_increment


        for i, distance in enumerate(ranges):


            if distance <= self.scan_data.range_min or distance >= self.scan_data.range_max:
                continue
            local_angle = angle_min + (i*angle_increment)


            if distance < self.distance_safe and (abs(local_angle) < math.radians(70)):
                angle = self.current_yaw + local_angle
                magnitude = self.k_obstacle*(1.0/distance**2.0)
                force_x -= magnitude * math.cos(angle)
                force_y -= magnitude * math.sin(angle)


        print(f"Force x is {str(force_x)}, Force y is {str(force_y)} for obstacles")
        return np.array([force_x, force_y])
   
    def force_goal(self, goal_x, goal_y):
        error_x = goal_x - self.current_x
        error_y = goal_y - self.current_y


        distance = math.hypot(error_x, error_y)
        if distance == 0.0:
            return np.array([0.0, 0.0])
       
        force_x = self.k_goal * (error_x/distance)
        force_y = self.k_goal * (error_y/distance)
        print(f"Force x is {str(force_x)}, Force y is {str(force_y)} for goals")
        return np.array([force_x, force_y])


    def navigation_control(self):
        if self.current_waypoint >= len(self.waypoints):
            self.get_logger().info('All zone visited!')
            self.on_shutdown()
            return
        print(f"Current waypoint goal {str(self.current_waypoint)}")


        goal_x, goal_y = self.waypoints[self.current_waypoint]
        distance_to_goal = math.hypot(goal_x - self.current_x, goal_y - self.current_y)


        # Check if the waypoint has been reached
        if distance_to_goal < self.goal_tolerance:
            self.get_logger().info(f"Reached waypoing {str(self.current_waypoint)}")
            self.current_waypoint += 1
            self.state = 'SEEK_GOAL'
            return


        msg = TwistStamped()


        if self.state == 'SEEK_GOAL':
            F_obstacle = self.force_obstacles()
            F_goal = self.force_goal(goal_x, goal_y)
            F_total = F_obstacle + F_goal


            angle_goal = math.atan2(F_goal[1], F_goal[0])
            angle_obstacles = math.atan2(F_obstacle[1], F_obstacle[0])
            magitude = math.hypot(F_obstacle[0], F_obstacle[1])


            print(f"Magnitude: {str(magitude)}")
            if magitude > 0.05 and (time.time() - self.last_escape_time > 3.0):
                angle_diff = abs(math.atan2(math.sin(angle_goal-angle_obstacles),math.cos(angle_goal-angle_obstacles)))
                print(f"angle diff {str(angle_diff)}")


                if angle_diff > 2.5:
                    self.get_logger().warn('Trap Detected! Switching to WALL_FOLLOW.')
                    self.state = 'WALL_FOLLOW'
                    self.distance_to_goal_at_trap = distance_to_goal
                    return
           


            desired_yaw = math.atan2(F_total[1], F_total[0]) - self.current_yaw
            yaw_error = math.atan2(math.sin(desired_yaw), math.cos(desired_yaw))
       
            print(f"Desired yaw is {str(desired_yaw)}, Yaw Error is {str(yaw_error)}")


            msg.twist.angular.z = 1.5 * yaw_error
            if abs(yaw_error) < 0.5:
                speed = math.hypot(F_total[1], F_total[0])
                # speed = 1.0
                msg.twist.linear.x = min(speed, self.max_speed)
                print("here")
            else:
                msg.twist.linear.x = 0.1
        elif self.state == 'WALL_FOLLOW':


            if distance_to_goal < (self.distance_to_goal_at_trap - 0.3):
                self.get_logger().info('Wall cleared! Resuming APF.')
                self.state = 'SEEK_GOAL'
                self.last_escape_time = time.time() # Start the cooldown timer
                return
           
            if not self.scan_data:
                return
           
            valid_ranges = [r for r in self.scan_data.ranges if self.scan_data.range_min < r < self.scan_data.range_max and not math.isinf(r)]


            if not valid_ranges:
                self.state = 'SEEK_GOAL'
                return
           
            dist = self.scan_data.ranges[:150]
            min_dist = min(valid_ranges)
            min_index = self.scan_data.ranges.index(min_dist)
            local_angle = self.scan_data.angle_min + (min_index * self.scan_data.angle_increment)


            target_angle = 1.57
            target_dist = self.distance_safe


            angle_error = local_angle - target_angle
            angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))
            dist_error = target_dist - min_dist
            derivative_dist = dist_error - self.prev_dist_error
            self.prev_dist_error = dist_error
           
            if abs(angle_error) > 0.5:
                msg.twist.linear.x = 0.0  # Stop forward movement! Just rotate in place.
                msg.twist.angular.z = 3.0 * angle_error
                self.get_logger().info('Aligning wall to the right side...')
            else:
                msg.twist.linear.x = 0.1  # Wall is safely on the side, move forward
                msg.twist.angular.z = (3.0 * angle_error) + (0.8 * dist_error) + (2.0 * derivative_dist)


        self.vel_pub.publish(msg)


    def on_shutdown(self):
        self.get_logger().info(
            "Stopping the robot..."
        )
        self.vel_pub.publish(TwistStamped())
        self.shutdown = True




def main(args=None):
    rclpy.init(
        args = args,
        signal_handler_options = SignalHandlerOptions.NO
    )
    node = AutonomousNavigation()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Shutdown request detected..")
    finally:
        node.on_shutdown()
        while not node.shutdown():
            continue
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

