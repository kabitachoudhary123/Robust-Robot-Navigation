#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from nav_msgs.msg import Odometry, Path, OccupancyGrid
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from tf_transformations import euler_from_quaternion
import math
import numpy as np
import random
import time
from typing import List, Tuple, Optional
import traceback
from scipy.interpolate import splprep, splev

# Constants
FREE_SPACE = 0
OCCUPIED_SPACE = 100
UNKNOWN_SPACE = -1

class MCTSNode:
    def __init__(self, state: Tuple[int, int], parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = []
        
    def is_fully_expanded(self):
        return len(self.untried_actions) == 0
        
    def best_child(self, exploration_weight=1.0):
        if not self.children:
            return None
            
        def uct(child):
            exploitation = child.value / (child.visits + 1e-6)
            exploration = exploration_weight * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6))
            return exploitation + exploration
            
        return max(self.children, key=uct)

class MCTSPlanner:
    def __init__(self, grid_width: int, grid_height: int, obstacles: List[Tuple[int, int]]):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.obstacles = set(obstacles)
        self.max_iterations = 10000
        self.max_simulation_steps = 1000
        self.exploration_weight = 1.0
        self.goal_reward = 1000.0
        self.step_penalty = -1.0
        self.collision_penalty = -500.0
        self.distance_penalty_factor = 0.5
        
    def is_valid_state(self, state: Tuple[int, int]) -> bool:
        x, y = state
        return (0 <= x < self.grid_width and 
                0 <= y < self.grid_height and 
                state not in self.obstacles)
    
    def get_actions(self, state: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = state
        possible_actions = [
            (x+1, y), (x-1, y), (x, y+1), (x, y-1),
            (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)
        ]
        return [a for a in possible_actions if self.is_valid_state(a)]
    
    def simulate(self, start: Tuple[int, int], goal: Tuple[int, int]) -> float:
        current = list(start)
        total_reward = 0.0
        steps = 0
        visited_in_sim = set()
        visited_in_sim.add(tuple(current))

        while tuple(current) != goal and steps < self.max_simulation_steps:
            steps += 1
            possible_next_states = self.get_actions(tuple(current))
            
            if not possible_next_states:
                total_reward += self.collision_penalty
                break
            
            valid_unvisited_next_states = [
                s for s in possible_next_states if s not in visited_in_sim
            ]
            
            states_for_choice = valid_unvisited_next_states if valid_unvisited_next_states else possible_next_states

            if states_for_choice:
                epsilon = 0.25
                if random.random() < epsilon:
                    next_state = random.choice(states_for_choice)
                else:
                    next_state = min(states_for_choice,
                                     key=lambda s: math.hypot(goal[0] - s[0], goal[1] - s[1]))
            else:
                total_reward += self.collision_penalty
                break
            
            current = list(next_state)
            visited_in_sim.add(tuple(current))
            total_reward += self.step_penalty
            distance_to_goal_now = math.hypot(goal[0] - current[0], goal[1] - current[1])
            total_reward -= distance_to_goal_now * self.distance_penalty_factor
            
        if tuple(current) == goal:
            total_reward += self.goal_reward
        else:
            final_distance_to_goal = math.hypot(goal[0] - current[0], goal[1] - current[1])
            total_reward -= final_distance_to_goal * (self.goal_reward / self.max_simulation_steps)

        return total_reward
    
    def search(self, root_state: Tuple[int, int], goal_state: Tuple[int, int]) -> List[Tuple[int, int]]:
        root_node = MCTSNode(root_state)
        root_node.untried_actions = self.get_actions(root_state)
        
        for _ in range(self.max_iterations):
            node = root_node
            state = root_state
            
            while node.is_fully_expanded() and node.children:
                node = node.best_child(self.exploration_weight)
                if node is None:
                    break

            if node is None:
                continue

            new_child_node = node
            if node.untried_actions:
                action = node.untried_actions.pop()
                new_state = action
                new_child_node = MCTSNode(new_state, parent=node)
                new_child_node.untried_actions = self.get_actions(new_state)
                node.children.append(new_child_node)
            
            reward = self.simulate(new_child_node.state, goal_state)
            
            current_node_for_backprop = new_child_node
            while current_node_for_backprop is not None:
                current_node_for_backprop.visits += 1
                current_node_for_backprop.value += reward
                current_node_for_backprop = current_node_for_backprop.parent
        
        path = []
        current_node = root_node
        path.append(current_node.state)

        while current_node and current_node.state != goal_state:
            best_child = current_node.best_child(exploration_weight=0)
            if best_child is None:
                break
            path.append(best_child.state)
            current_node = best_child
        
        if len(path) == 1 and path[0] != goal_state:
            return []

        return path

class MCTSPlannerNode(Node):
    def __init__(self):
        super().__init__('mcts_planner_node')
        
        # Initialize all variables
        self.robot_current_pose = None
        self.current_theta = None
        self.path_to_follow = []
        self.smoothed_path = []
        self.mcts_planner = None
        self.map_resolution = 0.05
        self.map_origin = [0.0, 0.0]
        self.occupancy_grid_data = None
        self.map_width = 0
        self.map_height = 0
        self.current_goal = None
        self.current_lookahead_point = None
        self.last_path_update = 0
        self.path_update_interval = 5.0
        self.pose_filter_alpha = 0.2
        self.last_pose_time = 0
        self.last_robot_position = None
        self.robot_speed = 0.0
        self.path_execution_start_time = 0
        self.visualization_index = 0
        self.last_visualization_time = 0
        
        # Declare parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('timer_period', 0.2),
                ('max_linear_vel', 0.2),
                ('min_linear_vel', 0.05),
                ('angular_vel_gain', 1.0),
                ('goal_tolerance', 0.1),
                ('waypoint_tolerance', 0.15),
                ('max_iterations', 10000),
                ('max_simulation_steps', 1000),
                ('exploration_weight', 1.0),
                ('replan_threshold', 1.0),
                ('smoothing_factor', 0.5),
                ('look_ahead_distance', 0.5),
                ('min_look_ahead', 0.3),
                ('max_look_ahead', 1.0),
                ('path_follow_gain', 0.5),
                ('max_speed_ratio', 0.8),
                ('visualization_delay', 0.2),
                ('speed_scaling_factor', 0.8),
                ('min_sync_distance', 0.3),
                ('sync_update_rate', 10.0),
                ('look_back_points', 3)
            ]
        )
        
        # Create subscribers
        self.amcl_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.amcl_pose_callback,
            10
        )
        
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            10
        )
        
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )
        
        # Create publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        self.smooth_path_pub = self.create_publisher(Path, '/smoothed_path', 10)
        self.debug_marker_pub = self.create_publisher(MarkerArray, '/debug_markers', 10)
        
        # Create timer
        timer_period = self.get_parameter('timer_period').get_parameter_value().double_value
        self.timer = self.create_timer(timer_period, self.plan_and_move_callback)
        
        self.get_logger().info("MCTS Planner Node initialized")

    def is_smoothed_path_clear(self, smoothed_path: List[Tuple[float, float]]) -> bool:
        """Check if smoothed path collides with obstacles"""
        if not self.mcts_planner:
            return True
            
        for wx, wy in smoothed_path:
            gx, gy = self.world_to_grid(wx, wy)
            if (gx, gy) in self.mcts_planner.obstacles:
                return False
                
        return True

    def smooth_path(self, path: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
        """Apply spline smoothing to the path with obstacle checking"""
        if len(path) < 3:
            return [self.grid_to_world(x, y) for x, y in path]
            
        # Original smoothing
        world_points = np.array([self.grid_to_world(x, y) for x, y in path])
        tck, u = splprep(world_points.T, s=self.get_parameter('smoothing_factor').value)
        u_new = np.linspace(0, 1, len(path)*3)
        x_new, y_new = splev(u_new, tck)
        smoothed = list(zip(x_new, y_new))
        
        # Verify path doesn't hit obstacles
        if not self.is_smoothed_path_clear(smoothed):
            self.get_logger().warn("Smoothed path hits obstacles, using original path")
            return [self.grid_to_world(x, y) for x, y in path]
            
        return smoothed

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
            
            self.mcts_planner = MCTSPlanner(
                grid_width=width,
                grid_height=height,
                obstacles=obstacles
            )
            
            if self.has_parameter('max_iterations'):
                self.mcts_planner.max_iterations = self.get_parameter('max_iterations').value
            if self.has_parameter('max_simulation_steps'):
                self.mcts_planner.max_simulation_steps = self.get_parameter('max_simulation_steps').value
            if self.has_parameter('exploration_weight'):
                self.mcts_planner.exploration_weight = self.get_parameter('exploration_weight').value
            
            self.get_logger().info(f"Map received. Size: {width}x{height}")
            
        except Exception as e:
            self.get_logger().error(f"Error processing map: {str(e)}")

    def amcl_pose_callback(self, msg: PoseWithCovarianceStamped):
        try:
            current_time = time.time()
            
            # Calculate robot speed
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
                    self.robot_current_pose.position.x = (1-alpha)*self.robot_current_pose.position.x + alpha*msg.pose.pose.position.x
                    self.robot_current_pose.position.y = (1-alpha)*self.robot_current_pose.position.y + alpha*msg.pose.pose.position.y
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
            self.path_execution_start_time = time.time()
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

    def plan_and_move_callback(self):
        try:
            if not self.smoothed_path or self.robot_current_pose is None or self.current_goal is None:
                return
                
            current_time = time.time()
            
            # Update visualization progress
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
            
            # Find closest point on smoothed path
            closest_idx = 0
            min_dist = float('inf')
            for i, (wx, wy) in enumerate(self.smoothed_path):
                dist = math.hypot(wx - current_wx, wy - current_wy)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
            
            # Stay slightly behind the closest point based on robot speed
            look_back_points = min(5, max(1, int(self.robot_speed * 2)))
            closest_idx = max(0, closest_idx - look_back_points)
            
            # Get lookahead point
            look_ahead_dist = self.get_parameter('look_ahead_distance').value
            min_look_ahead = self.get_parameter('min_look_ahead').value
            max_look_ahead = self.get_parameter('max_look_ahead').value
            
            # Dynamic lookahead based on speed and distance to goal
            speed_ratio = min(1.0, self.robot_speed / self.get_parameter('max_linear_vel').value)
            look_ahead_dist = min(max_look_ahead, max(min_look_ahead, 
                look_ahead_dist * (1 + speed_ratio * self.get_parameter('max_speed_ratio').value)))
            
            target_idx = closest_idx
            accumulated_dist = 0.0
            while target_idx < len(self.smoothed_path)-1 and accumulated_dist < look_ahead_dist:
                x1, y1 = self.smoothed_path[target_idx]
                x2, y2 = self.smoothed_path[target_idx+1]
                segment_length = math.hypot(x2-x1, y2-y1)
                accumulated_dist += segment_length
                target_idx += 1
            
            target_idx = min(target_idx, len(self.smoothed_path)-1)
            target_wx, target_wy = self.smoothed_path[target_idx]
            self.current_lookahead_point = (target_wx, target_wy)
            
            # Calculate control commands
            dx = target_wx - current_wx
            dy = target_wy - current_wy
            target_angle = math.atan2(dy, dx)
            angle_diff = (target_angle - self.current_theta + math.pi) % (2 * math.pi) - math.pi
            
            max_linear_vel = self.get_parameter('max_linear_vel').value
            min_linear_vel = self.get_parameter('min_linear_vel').value
            path_follow_gain = self.get_parameter('path_follow_gain').value
            angular_vel_gain = self.get_parameter('angular_vel_gain').value
            
            # Calculate base speed with synchronization to visualization
            base_speed = min(
                max_linear_vel * self.get_parameter('speed_scaling_factor').value,
                max(min_linear_vel, max_linear_vel * (1 - path_follow_gain * self.get_distance_to_path()))
            )
            
            # Ensure robot doesn't get too far ahead of visualization
            robot_dist_to_viz = self.get_distance_to_visualization()
            if robot_dist_to_viz > self.get_parameter('min_sync_distance').value:
                base_speed *= 0.7  # Slow down if getting ahead
            
            # Adjust speed based on angle difference
            if abs(angle_diff) > math.pi/6:
                linear_vel = min_linear_vel  # Slow down if angle is too large
                angular_vel = angle_diff * angular_vel_gain * 1.5  # Turn more aggressively
            elif abs(angle_diff) > math.pi/12:
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
        """Calculate distance between robot and current visualization point"""
        if not self.smoothed_path or not self.robot_current_pose or self.visualization_index >= len(self.smoothed_path):
            return float('inf')
            
        viz_x, viz_y = self.smoothed_path[self.visualization_index]
        robot_x = self.robot_current_pose.position.x
        robot_y = self.robot_current_pose.position.y
        
        return math.hypot(viz_x - robot_x, viz_y - robot_y)

    def update_visualization_progress(self):
        """Synchronize visualization with robot progress"""
        if not self.smoothed_path or not self.robot_current_pose:
            return
            
        # Find closest point on path to robot
        closest_idx = 0
        min_dist = float('inf')
        for i, (wx, wy) in enumerate(self.smoothed_path):
            dist = math.hypot(wx - self.robot_current_pose.position.x, 
                             wy - self.robot_current_pose.position.y)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # Set visualization slightly behind robot
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
            path_msg.poses.append(pose)
        
        self.path_pub.publish(path_msg)

    def publish_smoothed_path(self):
        path_msg = Path()
        path_msg.header = Header(
            stamp=self.get_clock().now().to_msg(),
            frame_id='map'
        )
        
        for i in range(min(self.visualization_index + 1, len(self.smoothed_path))):
            wx, wy = self.smoothed_path[i]
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            path_msg.poses.append(pose)
        
        self.smooth_path_pub.publish(path_msg)
        self.last_visualization_time = time.time()




        
        

    def publish_twist(self, linear_x: float, angular_z: float):
        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        self.cmd_vel_pub.publish(twist)

    def publish_debug_markers(self):
        marker_array = MarkerArray()
        
        # Goal marker
        if self.current_goal:
            goal_wx, goal_wy = self.current_goal
            
            marker = Marker()
            marker.header.frame_id = "map"
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.pose.position.x = goal_wx
            marker.pose.position.y = goal_wy
            marker.id = 0
            marker_array.markers.append(marker)
        
        # Lookahead point marker
        if self.current_lookahead_point:
            lap_wx, lap_wy = self.current_lookahead_point
            
            marker = Marker()
            marker.header.frame_id = "map"
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.pose.position.x = lap_wx
            marker.pose.position.y = lap_wy
            marker.id = 1
            marker_array.markers.append(marker)
        
        # Current pose marker
        if self.robot_current_pose and self.current_theta is not None:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.scale.x = 0.3
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.pose.position.x = self.robot_current_pose.position.x
            marker.pose.position.y = self.robot_current_pose.position.y
            marker.pose.orientation.z = math.sin(self.current_theta/2)
            marker.pose.orientation.w = math.cos(self.current_theta/2)
            marker.id = 2
            marker_array.markers.append(marker)
        
        self.debug_marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = MCTSPlannerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().error(f"Error in main: {str(e)}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()