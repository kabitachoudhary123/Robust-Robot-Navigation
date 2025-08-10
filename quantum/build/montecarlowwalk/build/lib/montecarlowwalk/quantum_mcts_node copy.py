#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from visualization_msgs.msg import Marker, MarkerArray
from tf_transformations import euler_from_quaternion
import math
import numpy as np
import time
from typing import List, Tuple, Optional
from scipy.interpolate import splprep, splev

# Quantum Computing Imports
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit.library import GroverOperator

# Constants
UNKNOWN_SPACE = -1
MAX_DEPTH_LIMIT = 500
MAX_QUANTUM_STATES = 16

class QuantumMCTSNode:
    def __init__(self, state: Tuple[int, int], parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = []
        self.quantum_boost = False

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, exploration_weight=1.0, use_quantum=False):
        if not self.children:
            return None
        return self.quantum_best_child(exploration_weight) if (use_quantum and len(self.children) > 3) else self.classical_best_child(exploration_weight)

    def classical_best_child(self, exploration_weight):
        def uct(child):
            return (child.value / (child.visits + 1e-6)) + (
                exploration_weight * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6))
            )
        return max(self.children, key=uct)

    def quantum_best_child(self, exploration_weight):
        try:
            num_children = len(self.children)
            if num_children > MAX_QUANTUM_STATES:
                return self.classical_best_child(exploration_weight)
                
            num_qubits = math.ceil(math.log2(num_children))
            values = [c.value / (c.visits + 1e-6) for c in self.children]
            threshold = np.percentile(values, 75)
            good_states = [i for i, v in enumerate(values) if v >= threshold]

            if not good_states or len(good_states) == num_children:
                return self.classical_best_child(exploration_weight)
            
            grover = GroverOperator(good_states, num_qubits=num_qubits)
            qc = QuantumCircuit(num_qubits)
            qc.h(range(num_qubits))
            qc.compose(grover, inplace=True)
            
            simulator = AerSimulator()
            result = simulator.run(qc, shots=100).result()
            counts = result.get_counts()
            
            selected_idx = int(max(counts.items(), key=lambda x: x[1])[0], 2)
            selected_idx = min(selected_idx, num_children-1)
            
            self.quantum_boost = True
            return self.children[selected_idx]
        except Exception:
            return self.classical_best_child(exploration_weight)

class QuantumMCTSPlanner:
    def __init__(self, get_logger_func, grid_width: int, grid_height: int, obstacles: List[Tuple[int, int]]):
        self.get_logger = get_logger_func
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.obstacles = set(obstacles)
        self.max_iterations = 1000
        self.max_simulation_steps = 200
        self.exploration_weight = 1.0
        self.goal_reward = 1000.0
        self.step_penalty = -1.0
        self.collision_penalty = -500.0
        self.distance_penalty_factor = 0.5
        self.quantum_threshold = 5
        self.current_goal = None
        self.simulation_cache = {}

    def is_valid_state(self, state: Tuple[int, int]) -> bool:
        x, y = state
        return (0 <= x < self.grid_width and 
                0 <= y < self.grid_height and 
                state not in self.obstacles)
    
    def heuristic_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    def get_actions(self, state: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = state
        possible_actions = [
            (x+1, y), (x-1, y), (x, y+1), (x, y-1),
            (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)
        ]
        valid_actions = [a for a in possible_actions if self.is_valid_state(a)]
        
        if self.current_goal:
            valid_actions.sort(key=lambda a: self.heuristic_distance(a, self.current_goal))
        
        return valid_actions

    def simulate(self, start: Tuple[int, int], goal: Tuple[int, int]) -> float:
        cache_key = (start, goal)
        if cache_key in self.simulation_cache:
            return self.simulation_cache[cache_key]
        
        current = start
        total_reward = 0.0
        visited = set([current])
        
        for step in range(self.max_simulation_steps):
            if current == goal:
                total_reward += self.goal_reward
                break
                
            possible_next = self.get_actions(current)
            if not possible_next:
                total_reward += self.collision_penalty
                break
                
            next_state = min(possible_next, key=lambda s: self.heuristic_distance(s, goal))
            current = next_state
            visited.add(current)
            total_reward += self.step_penalty
            total_reward -= self.heuristic_distance(current, goal) * self.distance_penalty_factor
        
        self.simulation_cache[cache_key] = total_reward
        return total_reward
    
    def search(self, root_state: Tuple[int, int], goal_state: Tuple[int, int]) -> List[Tuple[int, int]]:
        self.current_goal = goal_state
        self.simulation_cache = {}
        
        for depth_limit in [50, 100, 200, MAX_DEPTH_LIMIT]:
            path = self._limited_depth_search(root_state, goal_state, depth_limit)
            if path:
                return path
                
        self.get_logger().warn("Path not found within depth limits")
        return []

    def _limited_depth_search(self, root_state, goal_state, max_depth):
        root_node = QuantumMCTSNode(root_state)
        root_node.untried_actions = self.get_actions(root_state)
        
        for _ in range(min(self.max_iterations, max_depth)):
            node = root_node
            path = []
            
            # Selection
            while len(path) < max_depth and node.is_fully_expanded() and node.children:
                node = node.best_child(self.exploration_weight, len(path) % 10 == 0)
                path.append(node.state)
                if node.state == goal_state:
                    return [root_node.state] + path

            # Expansion
            if node.untried_actions and len(path) < max_depth:
                action = node.untried_actions.pop()
                new_child = QuantumMCTSNode(action, parent=node)
                new_child.untried_actions = self.get_actions(action)
                node.children.append(new_child)
                path.append(new_child.state)
                if new_child.state == goal_state:
                    return [root_node.state] + path

            # Simulation
            if path:
                reward = self.simulate(path[-1], goal_state)
                
                # Backpropagation
                current = new_child if 'new_child' in locals() else node
                backprop_steps = 0
                while current and backprop_steps < max_depth:
                    current.visits += 1
                    current.value += reward
                    current = current.parent
                    backprop_steps += 1

        # Extract best path
        path = []
        current = root_node
        while current and len(path) < max_depth:
            path.append(current.state)
            if current.state == goal_state:
                break
            current = current.best_child(exploration_weight=0)
            
        return path if len(path) > 1 else []

class QuantumMCTSPlannerNode(Node):
    def __init__(self):
        super().__init__('quantum_mcts_planner_node')
        self.mcts_planner = None
        self.robot_current_pose = None
        self.current_theta = 0.0
        self.current_goal = None
        self.path_to_follow = []
        self.smoothed_path = []
        self.visualization_index = 0
        self.last_visualization_time = time.time()
        self.last_path_update = 0
        self.path_update_interval = 2.0
        self.robot_speed = 0.0
        self.last_robot_position = None
        self.last_pose_time = 0
        self.pose_filter_alpha = 0.2
        self.current_lookahead_point = None
        self.map_resolution = 0.05
        self.map_origin = [0.0, 0.0]

        # Parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('max_linear_vel', 0.3),
                ('min_linear_vel', 0.05),
                ('max_angular_vel', 1.0),
                ('look_ahead_distance', 0.5),
                ('min_look_ahead', 0.3),
                ('max_look_ahead', 1.0),
                ('goal_tolerance', 0.1),
                ('replan_threshold', 0.3),
                ('path_follow_gain', 0.5),
                ('angular_vel_gain', 1.0),
                ('speed_scaling_factor', 1.0),
                ('max_speed_ratio', 0.5),
                ('min_sync_distance', 0.2),
                ('look_back_points', 3),
                ('sync_update_rate', 10.0)
            ]
        )

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, 'mcts_path', 10)
        self.smooth_path_pub = self.create_publisher(Path, 'smooth_path', 10)
        self.debug_marker_pub = self.create_publisher(MarkerArray, 'debug_markers', 10)

        # Subscribers
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            'map',
            self.map_callback,
            10
        )
        self.amcl_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            'amcl_pose',
            self.amcl_pose_callback,
            10
        )
        self.goal_sub = self.create_subscription(
            PoseStamped,
            'goal_pose',
            self.goal_callback,
            10
        )

        # Timer
        self.timer = self.create_timer(0.1, self.plan_and_move_callback)
    
    def map_callback(self, msg: OccupancyGrid):
        try:
            self.map_resolution = msg.info.resolution
            self.map_origin = [msg.info.origin.position.x, msg.info.origin.position.y]
            width = msg.info.width
            height = msg.info.height
            
            grid_data = np.array(msg.data).reshape((height, width))
            obstacles = [(x, y) for y in range(height) 
                         for x in range(width) 
                         if grid_data[y, x] > 50 or grid_data[y, x] == UNKNOWN_SPACE]
            
            self.mcts_planner = QuantumMCTSPlanner(self.get_logger, width, height, obstacles)
            self.get_logger().info(f"Quantum MCTS initialized with {len(obstacles)} obstacles")
            
        except Exception as e:
            self.get_logger().error(f"Map processing failed: {str(e)}")

    def amcl_pose_callback(self, msg: PoseWithCovarianceStamped):
        try:
            current_time = time.time()
            
            if self.last_robot_position is not None:
                dx = msg.pose.pose.position.x - self.last_robot_position[0]
                dy = msg.pose.pose.position.y - self.last_robot_position[1]
                dt = current_time - self.last_pose_time
                if dt > 0:
                    self.robot_speed = math.hypot(dx, dy) / dt
                else:
                    self.robot_speed = 0.0
            
            if self.robot_current_pose is None:
                self.robot_current_pose = msg.pose.pose
            else:
                if current_time > self.last_pose_time:
                    alpha = self.pose_filter_alpha
                    self.robot_current_pose.position.x = (1-alpha) * self.robot_current_pose.position.x + alpha * msg.pose.pose.position.x
                    self.robot_current_pose.position.y = (1-alpha) * self.robot_current_pose.position.y + alpha * msg.pose.pose.position.y
                    self.robot_current_pose.orientation = msg.pose.pose.orientation
            
            self.last_pose_time = current_time
            self.last_robot_position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
            
            orientation = self.robot_current_pose.orientation
            _, _, theta = euler_from_quaternion(
                [orientation.x, orientation.y, orientation.z, orientation.w]
            )
            self.current_theta = (theta + math.pi) % (2 * math.pi) - math.pi
            
        except Exception as e:
            self.get_logger().error(f"Error in AMCL pose callback: {str(e)}")

    def goal_callback(self, msg: PoseStamped):
        try:
            if self.mcts_planner is None:
                self.get_logger().warn("Map not yet received, ignoring goal")
                return
            
            goal_x = msg.pose.position.x
            goal_y = msg.pose.position.y
            self.current_goal = (goal_x, goal_y)
            self.last_path_update = time.time()
            self.visualization_index = 0
            
            self.get_logger().info(f"Received goal at: ({goal_x:.2f}, {goal_y:.2f})")
            
            goal_gx, goal_gy = self.world_to_grid(goal_x, goal_y)
            
            if not self.mcts_planner.is_valid_state((goal_gx, goal_gy)):
                self.get_logger().warn(f"Invalid goal position: ({goal_gx}, {goal_gy})")
                return
            
            if self.robot_current_pose is None:
                self.get_logger().warn("Robot pose not yet received")
                return
            
            current_gx, current_gy = self.world_to_grid(
                self.robot_current_pose.position.x,
                self.robot_current_pose.position.y
            )
            
            start_time = time.time()
            path = self.mcts_planner.search(
                root_state=(current_gx, current_gy),
                goal_state=(goal_gx, goal_gy)
            )
            planning_time = time.time() - start_time
            
            if not path:
                self.get_logger().warn("MCTS failed to find a path")
                return
            
            self.get_logger().info(f"Found path with {len(path)} points in {planning_time:.2f}s")
            self.path_to_follow = path
            self.smoothed_path = self.smooth_path(path)
            self.publish_path(path)
            self.publish_smoothed_path()
            
        except Exception as e:
            self.get_logger().error(f"Error in goal callback: {str(e)}")

    def smooth_path(self, path: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
        if len(path) < 3:
            return [self.grid_to_world(gx, gy) for gx, gy in path]
            
        world_path = [self.grid_to_world(gx, gy) for gx, gy in path]
        x = [p[0] for p in world_path]
        y = [p[1] for p in world_path]
        
        try:
            tck, u = splprep([x, y], s=2.0)
            u_new = np.linspace(0, 1, len(path)*3)
            x_new, y_new = splev(u_new, tck)
            return list(zip(x_new, y_new))
        except Exception as e:
            self.get_logger().warn(f"Path smoothing failed: {str(e)}")
            return world_path

    def plan_and_move_callback(self):
        try:
            if not self.smoothed_path or not self.robot_current_pose or not self.current_goal:
                return
            
            current_time = time.time()
            
            if current_time - self.last_visualization_time > (1.0 / self.get_parameter('sync_update_rate').value):
                self.update_visualization_progress()
                self.publish_smoothed_path()

            current_wx = self.robot_current_pose.position.x
            current_wy = self.robot_current_pose.position.y
            goal_wx, goal_wy = self.current_goal
            
            distance_to_goal = math.hypot(goal_wx - current_wx, goal_wy - current_wy)
            goal_tolerance = self.get_parameter('goal_tolerance').value
            
            if distance_to_goal < goal_tolerance:
                self.get_logger().info(f"Goal reached at: ({current_wx:.2f}, {current_wy:.2f})")
                self.path_to_follow = []
                self.smoothed_path = []
                self.publish_twist(0.0, 0.0)
                self.current_goal = None
                return
            
            replan_threshold = self.get_parameter('replan_threshold').value
            if (current_time - self.last_path_update > self.path_update_interval and
                self.get_distance_to_path() > replan_threshold):
                
                self.get_logger().info("Replanning path...")
                current_gx, current_gy = self.world_to_grid(current_wx, current_wy)
                goal_gx, goal_gy = self.world_to_grid(goal_wx, goal_wy)
                
                path = self.mcts_planner.search(
                    root_state=(current_gx, current_gy),
                    goal_state=(goal_gx, goal_gy)
                )
                
                if path:
                    self.path_to_follow = path
                    self.smoothed_path = self.smooth_path(path)
                    self.publish_path(path)
                    self.publish_smoothed_path()
                    self.last_path_update = current_time

            closest_idx = 0
            min_dist = float('inf')
            for i, (wx, wy) in enumerate(self.smoothed_path):
                dist = math.hypot(wx - current_wx, wy - current_wy)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
            
            look_back_points = min(5, max(1, int(self.robot_speed * 2)))
            closest_idx = max(0, closest_idx - look_back_points)
            
            look_ahead_dist = self.get_parameter('look_ahead_distance').value
            min_look_ahead = self.get_parameter('min_look_ahead').value
            max_look_ahead = self.get_parameter('max_look_ahead').value
            
            speed_ratio = min(1.0, self.robot_speed / self.get_parameter('max_linear_vel').value)
            look_ahead_dist = min(max_look_ahead, max(min_look_ahead, 
                                look_ahead_dist * (1 + speed_ratio * self.get_parameter('max_speed_ratio').value)))
            
            target_idx = closest_idx
            accumulated_dist = 0.0
            while target_idx < len(self.smoothed_path) - 1 and accumulated_dist < look_ahead_dist:
                x1, y1 = self.smoothed_path[target_idx]
                x2, y2 = self.smoothed_path[target_idx + 1]
                segment_length = math.hypot(x2 - x1, y2 - y1)
                accumulated_dist += segment_length
                target_idx += 1
            
            target_idx = min(target_idx, len(self.smoothed_path) - 1)
            target_wx, target_wy = self.smoothed_path[target_idx]
            self.current_lookahead_point = (target_wx, target_wy)
            
            dx = target_wx - current_wx
            dy = target_wy - current_wy
            target_angle = math.atan2(dy, dx)
            angle_diff = (target_angle - self.current_theta + math.pi) % (2 * math.pi) - math.pi
            
            max_linear_vel = self.get_parameter('max_linear_vel').value
            min_linear_vel = self.get_parameter('min_linear_vel').value
            path_follow_gain = self.get_parameter('path_follow_gain').value
            angular_vel_gain = self.get_parameter('angular_vel_gain').value
            
            base_speed = min(
                max_linear_vel * self.get_parameter('speed_scaling_factor').value,
                max(min_linear_vel, max_linear_vel * (1 - path_follow_gain * self.get_distance_to_path()))
            )
            
            robot_dist_to_viz = self.get_distance_to_visualization()
            if robot_dist_to_viz > self.get_parameter('min_sync_distance').value:
                base_speed *= 0.7
            
            if abs(angle_diff) > math.pi / 6:
                linear_vel = min_linear_vel
                angular_vel = angle_diff * angular_vel_gain * 1.5
            elif abs(angle_diff) > math.pi / 12:
                linear_vel = base_speed * 0.5
                angular_vel = angle_diff * angular_vel_gain
            else:
                linear_vel = base_speed
                angular_vel = angle_diff * angular_vel_gain
            
            self.publish_twist(linear_vel, angular_vel)
            self.publish_debug_markers()
            
        except Exception as e:
            self.get_logger().error(f"Error in plan_and_move_callback: {str(e)}")

    def get_distance_to_visualization(self) -> float:
        if not self.smoothed_path or not self.robot_current_pose or self.visualization_index >= len(self.smoothed_path):
            return float('inf')
            
        viz_x, viz_y = self.smoothed_path[self.visualization_index]
        robot_x = self.robot_current_pose.position.x
        robot_y = self.robot_current_pose.position.y
        
        return math.hypot(viz_x - robot_x, viz_y - robot_y)

    def update_visualization_progress(self):
        if not self.smoothed_path or not self.robot_current_pose:
            return
            
        closest_idx = 0
        min_dist = float('inf')
        for i, (wx, wy) in enumerate(self.smoothed_path):
            dist = math.hypot(wx - self.robot_current_pose.position.x, 
                              wy - self.robot_current_pose.position.y)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        look_back = max(1, int(self.get_parameter('look_back_points').value))
        self.visualization_index = max(0, closest_idx - look_back)
        
        self.last_visualization_time = time.time()

    def get_distance_to_path(self) -> float:
        if not self.smoothed_path or not self.robot_current_pose:
            return float('inf')
            
        current_wx = self.robot_current_pose.position.x
        current_wy = self.robot_current_pose.position.y
        
        min_distance = float('inf')
        for wx, wy in self.smoothed_path:
            distance = math.hypot(wx - current_wx, wy - current_wy)
            if distance < min_distance:
                min_distance = distance
                
        return min_distance

    def world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        gx = int(round((wx - self.map_origin[0]) / self.map_resolution))
        gy = int(round((wy - self.map_origin[1]) / self.map_resolution))
        if self.mcts_planner:
            gx = max(0, min(gx, self.mcts_planner.grid_width - 1))
            gy = max(0, min(gy, self.mcts_planner.grid_height - 1))
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        wx = self.map_origin[0] + (gx * self.map_resolution)
        wy = self.map_origin[1] + (gy * self.map_resolution)
        return wx, wy

    def publish_path(self, path: List[Tuple[int, int]]):
        path_msg = Path()
        path_msg.header = Header(
            stamp=self.get_clock().now().to_msg(),
            frame_id='map'
        )
        
        for gx, gy in path:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x, pose.pose.position.y = self.grid_to_world(gx, gy)
            pose.pose.orientation.w = 1.0  # Important for RViz
            path_msg.poses.append(pose)
        
        self.path_pub.publish(path_msg)
        self.get_logger().info(f"Published raw path with {len(path)} points")

    def publish_smoothed_path(self):
        if not self.smoothed_path:
            return
            
        path_msg = Path()
        path_msg.header = Header(
            stamp=self.get_clock().now().to_msg(),
            frame_id='map'
        )
        
        for wx, wy in self.smoothed_path:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.orientation.w = 1.0  # Important for RViz
            path_msg.poses.append(pose)
        
        self.smooth_path_pub.publish(path_msg)
        self.get_logger().info(f"Published smoothed path with {len(self.smoothed_path)} points")

    def publish_twist(self, linear_x: float, angular_z: float):
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = linear_x
        cmd_vel_msg.angular.z = angular_z
        self.cmd_vel_pub.publish(cmd_vel_msg)

    def publish_debug_markers(self):
        marker_array = MarkerArray()
        
        # Start marker
        if self.robot_current_pose:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = 0
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position = self.robot_current_pose.position
            marker.pose.position.z = 0.1
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker_array.markers.append(marker)
        
        # Goal marker
        if self.current_goal:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = 1
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = self.current_goal[0]
            marker.pose.position.y = self.current_goal[1]
            marker.pose.position.z = 0.1
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker_array.markers.append(marker)
        
        # Lookahead point
        if self.current_lookahead_point:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = 2
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = self.current_lookahead_point[0]
            marker.pose.position.y = self.current_lookahead_point[1]
            marker.pose.position.z = 0.1
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker_array.markers.append(marker)
        
        self.debug_marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    quantum_mcts_node = QuantumMCTSPlannerNode()
    rclpy.spin(quantum_mcts_node)
    quantum_mcts_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()