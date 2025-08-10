#!/usr/bin/env python3

# quantum_mcts_planner_node.py

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from std_msgs.msg import Header
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped, Pose
from visualization_msgs.msg import Marker, MarkerArray
from tf_transformations import euler_from_quaternion

import numpy as np
import math
import time
from typing import List, Tuple, Optional

# --- Import the core algorithm ---
from .quantum_mcts_core import QuantumMCTSPlanner

# Constants
UNKNOWN_SPACE = -1

class QuantumMCTSPlannerNode(Node):
    """The ROS2 Node for running the Quantum MCTS Planner."""
    def __init__(self):
        super().__init__('quantum_mcts_planner_node')
        self.mcts_planner: Optional[QuantumMCTSPlanner] = None
        self.robot_current_pose: Optional[Pose] = None
        self.current_theta: float = 0.0
        self.current_goal: Optional[Tuple[float, float]] = None
        self.path_to_follow: List[Tuple[int, int]] = []
        
        self.last_path_update: float = 0.0
        self.path_update_interval: float = 5.0
        self.current_lookahead_point: Optional[Tuple[float, float]] = None
        self.map_resolution: float = 0.0
        self.map_origin: List[float] = [0.0, 0.0]

        # Declare parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('max_linear_vel', 0.22), ('min_linear_vel', 0.05),
                ('max_angular_vel', 1.0), ('look_ahead_distance', 0.5),
                ('goal_tolerance', 0.15), ('replan_threshold', 0.5),
                ('angular_vel_gain', 1.2)
            ]
        )

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, 'mcts_path', 10)
        self.debug_marker_pub = self.create_publisher(MarkerArray, 'debug_markers', 10)

        # Subscribers with reliable QoS for map
        map_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_sub = self.create_subscription(OccupancyGrid, 'map', self.map_callback, map_qos)
        self.amcl_pose_sub = self.create_subscription(PoseWithCovarianceStamped, 'amcl_pose', self.amcl_pose_callback, 10)
        self.goal_sub = self.create_subscription(PoseStamped, 'goal_pose', self.goal_callback, 10)

        # Main control loop Timer
        self.timer = self.create_timer(0.1, self.plan_and_move_callback)
        self.get_logger().info("Quantum MCTS Planner Node has started and is waiting for a map.")

    def map_callback(self, msg: OccupancyGrid):
        if self.mcts_planner is not None:
            return  # Avoid re-initializing
        try:
            self.map_resolution = msg.info.resolution
            self.map_origin = [msg.info.origin.position.x, msg.info.origin.position.y]
            width, height = msg.info.width, msg.info.height
            
            grid_data = np.array(msg.data).reshape((height, width))
            obstacles = [(x, y) for y in range(height) for x in range(width) 
                         if grid_data[y, x] > 50 or grid_data[y, x] == UNKNOWN_SPACE]
            
            self.mcts_planner = QuantumMCTSPlanner(self.get_logger, width, height, obstacles)
            self.get_logger().info(f"Map received. MCTS core initialized with {len(obstacles)} obstacles.")
        except Exception as e:
            self.get_logger().error(f"Map processing failed: {e}")

    def amcl_pose_callback(self, msg: PoseWithCovarianceStamped):
        self.robot_current_pose = msg.pose.pose
        _, _, self.current_theta = euler_from_quaternion([
            self.robot_current_pose.orientation.x, self.robot_current_pose.orientation.y,
            self.robot_current_pose.orientation.z, self.robot_current_pose.orientation.w
        ])

    def goal_callback(self, msg: PoseStamped):
        if self.mcts_planner is None:
            self.get_logger().warn("Planner not ready: Map not yet received.")
            return
        if self.robot_current_pose is None:
            self.get_logger().warn("Planner not ready: Robot pose not yet received.")
            return

        self.current_goal = (msg.pose.position.x, msg.pose.position.y)
        self.last_path_update = time.time()
        self.get_logger().info(f"Received new goal at: ({self.current_goal[0]:.2f}, {self.current_goal[1]:.2f})")
        
        goal_gx, goal_gy = self.world_to_grid(*self.current_goal)
        if not self.mcts_planner.is_valid_state((goal_gx, goal_gy)):
            self.get_logger().error(f"Invalid Goal: Position ({goal_gx}, {goal_gy}) is in an obstacle.")
            self.current_goal = None
            return
            
        current_gx, current_gy = self.world_to_grid(self.robot_current_pose.position.x, self.robot_current_pose.position.y)
        
        self.get_logger().info("Starting MCTS search...")
        start_time = time.time()
        path = self.mcts_planner.search(root_state=(current_gx, current_gy), goal_state=(goal_gx, goal_gy))
        planning_time = time.time() - start_time
        
        if not path:
            self.get_logger().warn(f"MCTS failed to find a path in {planning_time:.2f}s.")
            self.current_goal = None
            return
        
        self.get_logger().info(f"Path found in {planning_time:.2f}s.")
        self.path_to_follow = path
        self.publish_path(path)
    
    def plan_and_move_callback(self):
        if not self.path_to_follow or self.robot_current_pose is None or self.current_goal is None:
            self.publish_twist(0.0, 0.0) # Ensure robot stops if there's no path
            return

        try:
            current_wx, current_wy = self.robot_current_pose.position.x, self.robot_current_pose.position.y
            goal_wx, goal_wy = self.current_goal

            if math.hypot(goal_wx - current_wx, goal_wy - current_wy) < self.get_parameter('goal_tolerance').value:
                self.get_logger().info("Goal reached!")
                self.path_to_follow, self.current_goal = [], None
                return

            closest_idx = min(range(len(self.path_to_follow)), key=lambda i: math.hypot(
                self.grid_to_world(*self.path_to_follow[i])[0] - current_wx,
                self.grid_to_world(*self.path_to_follow[i])[1] - current_wy
            ))

            look_ahead_dist = self.get_parameter('look_ahead_distance').value
            target_idx = closest_idx
            while target_idx < len(self.path_to_follow) - 1:
                wx, wy = self.grid_to_world(*self.path_to_follow[target_idx])
                if math.hypot(wx - current_wx, wy - current_wy) > look_ahead_dist:
                    break
                target_idx += 1
            
            target_wx, target_wy = self.grid_to_world(*self.path_to_follow[target_idx])
            self.current_lookahead_point = (target_wx, target_wy)
            
            angle_to_target = math.atan2(target_wy - current_wy, target_wx - current_wx)
            angle_diff = (angle_to_target - self.current_theta + math.pi) % (2 * math.pi) - math.pi
            
            max_linear = self.get_parameter('max_linear_vel').value
            max_angular = self.get_parameter('max_angular_vel').value
            
            angle_reduction = 1.0 - 0.8 * (abs(angle_diff) / math.pi)
            linear_vel = max(self.get_parameter('min_linear_vel').value, max_linear * angle_reduction)
            angular_vel = np.clip(angle_diff * self.get_parameter('angular_vel_gain').value, -max_angular, max_angular)

            self.publish_twist(linear_vel, angular_vel)
            self.publish_debug_markers()
        except Exception as e:
            self.get_logger().error(f"Error in plan_and_move_callback: {e}")

    def world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        if self.mcts_planner is None or self.map_resolution == 0: return (0, 0)
        gx = int((wx - self.map_origin[0]) / self.map_resolution)
        gy = int((wy - self.map_origin[1]) / self.map_resolution)
        return max(0, min(gx, self.mcts_planner.grid_width - 1)), max(0, min(gy, self.mcts_planner.grid_height - 1))

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        wx = self.map_origin[0] + (gx + 0.5) * self.map_resolution
        wy = self.map_origin[1] + (gy + 0.5) * self.map_resolution
        return wx, wy

    def publish_path(self, path: List[Tuple[int, int]]):
        path_msg = Path()
        path_msg.header.stamp, path_msg.header.frame_id = self.get_clock().now().to_msg(), 'map'
        path_msg.poses = [PoseStamped(header=path_msg.header, pose=Pose(position=Pose().position.__class__(x=wx, y=wy), orientation=Pose().orientation.__class__(w=1.0))) for wx, wy in (self.grid_to_world(gx, gy) for gx, gy in path)]
        self.path_pub.publish(path_msg)

    def publish_twist(self, linear_x: float, angular_z: float):
        self.cmd_vel_pub.publish(Twist(linear=Twist().linear.__class__(x=linear_x), angular=Twist().angular.__class__(z=angular_z)))

    def publish_debug_markers(self):
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()
        if self.current_goal:
            marker = Marker(header=Header(frame_id="map", stamp=now), ns="goal", id=1, type=Marker.SPHERE, action=Marker.ADD)
            marker.pose.position.x, marker.pose.position.y = self.current_goal
            marker.scale.x, marker.scale.y, marker.scale.z = 0.3, 0.3, 0.3
            marker.color.r, marker.color.a = 1.0, 1.0
            marker_array.markers.append(marker)
        if self.current_lookahead_point:
            marker = Marker(header=Header(frame_id="map", stamp=now), ns="lookahead", id=2, type=Marker.SPHERE, action=Marker.ADD)
            marker.pose.position.x, marker.pose.position.y = self.current_lookahead_point
            marker.scale.x, marker.scale.y, marker.scale.z = 0.2, 0.2, 0.2
            marker.color.b, marker.color.a = 1.0, 1.0
            marker_array.markers.append(marker)
        self.debug_marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = QuantumMCTSPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()