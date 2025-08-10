#!/usr/bin/env python3
# quantum_mcts_planner_node.py

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Header
from nav_msgs.msg import Path, OccupancyGrid
from nav2_msgs.action import ComputePathToPose
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped, Pose
from tf_transformations import euler_from_quaternion
import numpy as np
import math
import time
from typing import List, Tuple, Optional

from .quantum_mcts_core import QuantumMCTSPlanner

UNKNOWN_SPACE = -1

class QuantumMCTSPlannerNode(Node):
    def __init__(self):
        super().__init__('quantum_mcts_planner_node')
        self.mcts_planner: Optional[QuantumMCTSPlanner] = None
        self.robot_current_pose: Optional[Pose] = None
        self.current_theta: float = 0.0
        self.current_goal: Optional[Tuple[float, float]] = None
        self.path_to_follow: List[Tuple[int, int]] = []
        self.map_resolution: float = 0.0
        self.map_origin: List[float] = [0.0, 0.0]
        self.map_width: int = 0
        self.map_height: int = 0
        self.current_path_index: int = 0

        self.declare_parameters(
            namespace='',
            parameters=[
                ('max_linear_vel', 0.22), ('min_linear_vel', 0.05),
                ('max_angular_vel', 1.0), ('waypoint_tolerance', 0.2),
                ('goal_tolerance', 0.15), ('inflation_radius', 0.25),
                ('heuristic_weight', 15.0), 
                ('angular_vel_gain', 1.8),
                ('turn_in_place_threshold', 0.7),
                ('iterations_per_meter', 2500),
                ('min_iterations', 5000),
                ('max_iterations', 60000),
                ('max_planning_time', 15.0),
                ('distance_penalty_factor', 0.5)
            ]
        )
        
        # --- NEW: Action Server for the benchmark runner to call ---
        self._action_server = ActionServer(
            self,
            ComputePathToPose,
            'compute_path_to_pose_qmcts',
            execute_callback=self.execute_plan_callback,
            goal_callback=self.goal_check_callback
        )

        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, 'mcts_path', 10)
        self.inflated_map_pub = self.create_publisher(OccupancyGrid, 'inflated_map', QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        
        map_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_sub = self.create_subscription(OccupancyGrid, 'map', self.map_callback, map_qos)
        self.amcl_pose_sub = self.create_subscription(PoseWithCovarianceStamped, 'amcl_pose', self.amcl_pose_callback, 10)
        self.goal_sub = self.create_subscription(PoseStamped, 'goal_pose', self.rviz_goal_callback, 10)
        
        self.timer = self.create_timer(0.1, self.plan_and_move_callback)
        self.get_logger().info("Quantum MCTS Planner Node and Action Server started.")

    def goal_check_callback(self, goal_request):
        """Accepts or rejects a new goal."""
        if not self.mcts_planner or not self.robot_current_pose:
            self.get_logger().warn("Planner not ready, rejecting goal.")
            return GoalResponse.REJECT
        self.get_logger().info("New planning goal received, accepting.")
        return GoalResponse.ACCEPT

    def execute_plan_callback(self, goal_handle):
        """Executes the planning action."""
        start_pose = self.robot_current_pose
        goal_pose = goal_handle.request.goal.pose

        self.get_logger().info("Executing plan...")
        
        start_grid = self.world_to_grid(start_pose.position.x, start_pose.position.y)
        goal_grid = self.world_to_grid(goal_pose.position.x, goal_pose.position.y)

        distance_to_goal = math.hypot(goal_pose.position.x - start_pose.position.x, goal_pose.position.y - start_pose.position.y)
        iters_per_meter = self.get_parameter('iterations_per_meter').value
        min_iters, max_iters = self.get_parameter('min_iterations').value, self.get_parameter('max_iterations').value
        
        if iters_per_meter <= 0: iters_per_meter = 2500
        
        distance_based_iters = int(distance_to_goal * iters_per_meter)
        dynamic_iterations = max(min_iters, min(distance_based_iters, max_iters))
        
        max_plan_time = self.get_parameter('max_planning_time').value
        self.get_logger().info(f"Goal distance: {distance_to_goal:.2f}m. Using {dynamic_iterations} iterations with a {max_plan_time}s time limit.")
        
        start_time = time.time()
        path = self.mcts_planner.search(start_grid, goal_grid, dynamic_iterations, max_plan_time)
        planning_time = time.time() - start_time
        
        result = ComputePathToPose.Result()
        if not path:
            self.get_logger().warn(f"MCTS failed to find a path in {planning_time:.2f}s.")
            goal_handle.abort()
            return result
        
        smoothed_path = self.smooth_path(path)
        self.get_logger().info(f"Path found in {planning_time:.2f}s. Original: {len(path)} pts, Smoothed: {len(smoothed_path)} pts.")
        
        # Populate the result message for the action client
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'
        path_msg.poses = [PoseStamped(header=path_msg.header, pose=Pose(position=Pose().position.__class__(x=wx, y=wy), orientation=Pose().orientation.__class__(w=1.0))) for wx, wy in (self.grid_to_world(gx, gy) for gx, gy in smoothed_path)]
        result.path = path_msg
        
        goal_handle.succeed()
        
        # Set the path for this node to follow
        self.path_to_follow = smoothed_path
        self.current_path_index = 0
        self.current_goal = (goal_pose.position.x, goal_pose.position.y)
        self.publish_path(smoothed_path)

        return result

    def rviz_goal_callback(self, msg: PoseStamped):
        """Handles goals from RViz, but now just sets the internal goal state."""
        self.get_logger().info("Received goal from RViz. The action server will now handle planning.")
        # The action server's execute_plan_callback will be triggered by the benchmark runner
        # or a future client. For now, we just set the goal for the follower.
        # In a full system, you might have this callback also call the action server.
        pass

    def plan_and_move_callback(self):
        if not self.path_to_follow or not self.robot_current_pose or not self.current_goal:
            self.publish_twist(0.0, 0.0)
            return
        if self.current_path_index >= len(self.path_to_follow): return

        curr_pos = (self.robot_current_pose.position.x, self.robot_current_pose.position.y)
        target_grid_point = self.path_to_follow[self.current_path_index]
        target_world_point = self.grid_to_world(*target_grid_point)
        dist_to_target = math.hypot(target_world_point[0] - curr_pos[0], target_world_point[1] - curr_pos[1])
        
        is_last_waypoint = (self.current_path_index == len(self.path_to_follow) - 1)
        if is_last_waypoint and dist_to_target < self.get_parameter('goal_tolerance').value:
            self.get_logger().info("Goal reached!")
            self.path_to_follow, self.current_goal = [], None
            self.publish_twist(0.0, 0.0)
            return

        if not is_last_waypoint and dist_to_target < self.get_parameter('waypoint_tolerance').value:
            self.get_logger().info(f"Reached waypoint {self.current_path_index}, switching to next.")
            self.current_path_index += 1
            if self.current_path_index >= len(self.path_to_follow): return
            target_grid_point = self.path_to_follow[self.current_path_index]
            target_world_point = self.grid_to_world(*target_grid_point)

        angle_to_target = math.atan2(target_world_point[1] - curr_pos[1], target_world_point[0] - curr_pos[0])
        angle_diff = (angle_to_target - self.current_theta + math.pi) % (2 * math.pi) - math.pi
        
        max_linear, min_linear = self.get_parameter('max_linear_vel').value, self.get_parameter('min_linear_vel').value
        max_angular, angular_gain = self.get_parameter('max_angular_vel').value, self.get_parameter('angular_vel_gain').value
        turn_threshold = self.get_parameter('turn_in_place_threshold').value
        ang_vel = np.clip(angle_diff * angular_gain, -max_angular, max_angular)
        lin_vel = min_linear if abs(angle_diff) > turn_threshold else max_linear
        self.publish_twist(lin_vel, ang_vel)

    def map_callback(self, msg: OccupancyGrid):
        if self.mcts_planner is not None: return
        self.get_logger().info("Map received, processing for obstacle inflation...")
        self.map_resolution = msg.info.resolution
        self.map_origin = [msg.info.origin.position.x, msg.info.origin.position.y]
        self.map_width, self.map_height = msg.info.width, msg.info.height
        grid_data = np.array(msg.data, dtype=np.int8).reshape((self.map_height, self.map_width))
        obstacles = set((x, y) for y, x in np.argwhere((grid_data > 50) | (grid_data == UNKNOWN_SPACE)))
        inflation_radius_m = self.get_parameter('inflation_radius').value
        inflation_cells = math.ceil(inflation_radius_m / self.map_resolution)
        inflated_obstacles = set(obstacles)
        for ox, oy in obstacles:
            for dx in range(-inflation_cells, inflation_cells + 1):
                for dy in range(-inflation_cells, inflation_cells + 1):
                    if dx*dx + dy*dy <= inflation_cells*inflation_cells:
                        inflated_obstacles.add((ox + dx, oy + dy))
        
        heuristic_weight = self.get_parameter('heuristic_weight').value
        dist_penalty = self.get_parameter('distance_penalty_factor').value
        self.mcts_planner = QuantumMCTSPlanner(self.get_logger, self.map_width, self.map_height, inflated_obstacles, heuristic_weight, dist_penalty)
        
        self.get_logger().info(f"Planner initialized. Heuristic Weight: {heuristic_weight}, Inflation Radius: {inflation_radius_m}m.")
        self.publish_inflated_map(grid_data, inflated_obstacles)

    def publish_inflated_map(self, original_grid, inflated_obstacles):
        inflated_grid = np.copy(original_grid)
        for x, y in inflated_obstacles:
            if 0 <= y < self.map_height and 0 <= x < self.map_width: inflated_grid[y, x] = 100
        msg = OccupancyGrid(header=Header(stamp=self.get_clock().now().to_msg(), frame_id="map"))
        msg.info = OccupancyGrid().info.__class__(resolution=self.map_resolution, width=self.map_width, height=self.map_height, origin=Pose(position=Pose().position.__class__(x=self.map_origin[0], y=self.map_origin[1])))
        msg.data = inflated_grid.flatten().tolist()
        self.inflated_map_pub.publish(msg)

    def amcl_pose_callback(self, msg: PoseWithCovarianceStamped):
        self.robot_current_pose = msg.pose.pose
        _, _, self.current_theta = euler_from_quaternion([p for p in (self.robot_current_pose.orientation.x, self.robot_current_pose.orientation.y, self.robot_current_pose.orientation.z, self.robot_current_pose.orientation.w)])

    def smooth_path(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if len(path) < 3: return path
        def has_line_of_sight(p1, p2):
            x0, y0 = p1; x1, y1 = p2
            dx, dy = abs(x1 - x0), abs(y1 - y0)
            sx, sy = 1 if x0 < x1 else -1, 1 if y0 < y1 else -1
            err = dx - dy
            while True:
                if (x0, y0) in self.mcts_planner.obstacles: return False
                if x0 == x1 and y0 == y1: break
                e2 = 2 * err
                if e2 > -dy: err -= dy; x0 += sx
                if e2 < dx: err += dx; y0 += sy
            return True
        smoothed_path = [path[0]]
        curr_idx = 0
        while curr_idx < len(path) - 1:
            best_sight_idx = curr_idx + 1
            for next_idx in range(curr_idx + 2, len(path)):
                if has_line_of_sight(path[curr_idx], path[next_idx]): best_sight_idx = next_idx
                else: break
            smoothed_path.append(path[best_sight_idx])
            curr_idx = best_sight_idx
        return smoothed_path

    def world_to_grid(self, wx, wy):
        if not self.mcts_planner or self.map_resolution == 0: return (0, 0)
        gx = int((wx - self.map_origin[0]) / self.map_resolution)
        gy = int((wy - self.map_origin[1]) / self.map_resolution)
        return max(0, min(gx, self.map_width - 1)), max(0, min(gy, self.map_height - 1))

    def grid_to_world(self, gx, gy):
        return self.map_origin[0] + (gx + 0.5) * self.map_resolution, self.map_origin[1] + (gy + 0.5) * self.map_resolution

    def publish_path(self, path):
        path_msg = Path(header=Header(stamp=self.get_clock().now().to_msg(), frame_id='map'))
        path_msg.poses = [PoseStamped(header=path_msg.header, pose=Pose(position=Pose().position.__class__(x=wx, y=wy), orientation=Pose().orientation.__class__(w=1.0))) for wx, wy in (self.grid_to_world(gx, gy) for gx, gy in path)]
        self.path_pub.publish(path_msg)

    def publish_twist(self, x, z):
        self.cmd_vel_pub.publish(Twist(linear=Twist().linear.__class__(x=x), angular=Twist().angular.__class__(z=z)))

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
