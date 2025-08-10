#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from nav_msgs.msg import Odometry, Path, OccupancyGrid
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped, Point, Pose
from visualization_msgs.msg import Marker, MarkerArray
from tf_transformations import euler_from_quaternion
import math
import numpy as np
import random
import time
from typing import List, Tuple, Optional
import traceback
from threading import Lock, Thread

from scipy.ndimage import distance_transform_edt

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
    def __init__(self, grid_width: int, grid_height: int, obstacles: List[Tuple[int, int]], costmap: np.ndarray):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.obstacles = set(obstacles)
        self.costmap = costmap
        self.max_iterations = 75000
        self.max_simulation_steps = 1000
        self.exploration_weight = 1.0
        self.goal_reward = 1000.0
        self.step_penalty = -1.0
        self.collision_penalty = -500.0
        self.distance_penalty_factor = 0.5
        self.obstacle_proximity_penalty_factor = 10.0
        self.robot_radius = 0.25

    def is_valid_state(self, state: Tuple[int, int]) -> bool:
        x, y = state
        if not (0 <= x < self.grid_width and 0 <= y < self.grid_height):
            return False
        return state not in self.obstacles and self.costmap[y, x] > self.robot_radius
    
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
            
            if random.random() < 0.25:
                next_state = random.choice(possible_next_states)
            else:
                def heuristic(s):
                    dist_to_goal = math.hypot(goal[0] - s[0], goal[1] - s[1])
                    cost_from_obstacle = self.obstacle_proximity_penalty_factor / (self.costmap[s[1], s[0]] + 1e-6)
                    return dist_to_goal + cost_from_obstacle
                
                next_state = min(possible_next_states, key=heuristic)
            
            current = list(next_state)
            visited_in_sim.add(tuple(current))
            total_reward += self.step_penalty
            proximity_penalty = self.obstacle_proximity_penalty_factor / (self.costmap[current[1], current[0]] + 1e-6)
            total_reward -= proximity_penalty
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
        
        for iteration in range(self.max_iterations):
            node = root_node
            
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
        
        self.robot_current_pose = None
        self.current_theta = None
        self.mcts_planner = None
        self.map_resolution = 0.05
        self.map_origin = [0.0, 0.0]
        self.map_width = 0
        self.map_height = 0
        self.current_goal = None
        self.current_lookahead_point = None
        
        self.path_buffer = [] 
        self.execution_index = 0
        self.path_lock = Lock()
        self.planner_thread = None
        self.planner_active_flag = False
        
        self.declare_parameters(
            namespace='',
            parameters=[
                ('timer_period', 0.1),
                ('max_linear_vel', 0.22),
                ('min_linear_vel', 0.05),
                ('max_angular_vel', 1.5),
                ('angular_vel_gain', 1.2),
                ('cte_gain', 2.0),
                ('goal_tolerance', 0.15),
                ('look_ahead_distance', 0.4),
                ('mcts_max_iterations', 15000), # Reduced for faster chunking
                ('mcts_max_simulation_steps', 500), # Reduced for faster chunking
                ('mcts_exploration_weight', 2.0),
                ('path_chunk_length', 5.0) # New parameter for chunking
            ]
        )
        
        self.amcl_pose_sub = self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.amcl_pose_callback, 10)
        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, rclpy.qos.QoSProfile(depth=1, reliability=rclpy.qos.ReliabilityPolicy.RELIABLE, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL))

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        self.debug_marker_pub = self.create_publisher(MarkerArray, '/debug_markers', 10)
        
        timer_period = self.get_parameter('timer_period').get_parameter_value().double_value
        self.timer = self.create_timer(timer_period, self.plan_and_move_callback)
        
        self.get_logger().info("MCTS Planner Node initialized.")

    def map_callback(self, msg: OccupancyGrid):
        try:
            self.map_resolution = msg.info.resolution
            self.map_origin = [msg.info.origin.position.x, msg.info.origin.position.y]
            self.map_width = msg.info.width
            self.map_height = msg.info.height
            
            grid_data = np.array(msg.data).reshape((self.map_height, self.map_width))
            
            binary_grid = np.ones_like(grid_data, dtype=np.uint8)
            binary_grid[grid_data > 50] = 0
            binary_grid[grid_data == UNKNOWN_SPACE] = 0

            costmap = distance_transform_edt(binary_grid) * self.map_resolution
            
            obstacles = [(x, y) for y in range(self.map_height) 
                          for x in range(self.map_width) 
                          if grid_data[y, x] > 50 or grid_data[y, x] == UNKNOWN_SPACE]
            
            self.mcts_planner = MCTSPlanner(
                grid_width=self.map_width,
                grid_height=self.map_height,
                obstacles=obstacles,
                costmap=costmap
            )
            
            self.mcts_planner.max_iterations = self.get_parameter('mcts_max_iterations').value
            self.mcts_planner.max_simulation_steps = self.get_parameter('mcts_max_simulation_steps').value
            self.mcts_planner.exploration_weight = self.get_parameter('mcts_exploration_weight').value
            
            self.get_logger().info(f"Map and costmap received. Size: {self.map_width}x{self.map_height}")
            
        except Exception as e:
            self.get_logger().error(f"Error processing map: {str(e)}\n{traceback.format_exc()}")

    def amcl_pose_callback(self, msg: PoseWithCovarianceStamped):
        self.robot_current_pose = msg.pose.pose
        orientation = self.robot_current_pose.orientation
        _, _, theta = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        self.current_theta = theta 

    def goal_callback(self, msg: PoseStamped):
        if self.mcts_planner is None:
            self.get_logger().warn("Map not yet received, ignoring goal.")
            return

        self.current_goal = (msg.pose.position.x, msg.pose.position.y)
        self.execution_index = 0
        with self.path_lock:
            self.path_buffer = []

        self.get_logger().info(f"Received final goal: ({self.current_goal[0]:.2f}, {self.current_goal[1]:.2f})")
        
        self.plan_next_path_chunk()

    def _plan_next_path_chunk_async(self):
        try:
            self.planner_active_flag = True
            
            if self.robot_current_pose is None or self.current_goal is None:
                self.get_logger().warn("Cannot plan: robot pose or final goal is missing.")
                return

            current_wx = self.robot_current_pose.position.x
            current_wy = self.robot_current_pose.position.y
            goal_wx, goal_wy = self.current_goal
            
            current_gx, current_gy = self.world_to_grid(current_wx, current_wy)
            goal_gx, goal_gy = self.world_to_grid(goal_wx, goal_wy)

            if not self.mcts_planner.is_valid_state((goal_gx, goal_gy)):
                self.get_logger().warn(f"Invalid or unsafe goal position: ({goal_gx}, {goal_gy}). Ignoring goal.")
                self.current_goal = None
                return
            
            dist_to_final_goal_cells = math.hypot(goal_gx - current_gx, goal_gy - current_gy)
            path_chunk_length = self.get_parameter('path_chunk_length').value
            chunk_length_cells = int(path_chunk_length / self.map_resolution)
            
            if dist_to_final_goal_cells < chunk_length_cells:
                sub_goal_gx, sub_goal_gy = goal_gx, goal_gy
                self.get_logger().info("Planning final chunk to the goal.")
            else:
                vector_x = (goal_gx - current_gx) / dist_to_final_goal_cells
                vector_y = (goal_gy - current_gy) / dist_to_final_goal_cells
                sub_goal_gx = int(current_gx + vector_x * chunk_length_cells)
                sub_goal_gy = int(current_gy + vector_y * chunk_length_cells)
                self.get_logger().info(f"Planning new path chunk to sub-goal: ({sub_goal_gx}, {sub_goal_gy})")

            start_time = time.time()
            path_chunk = self.mcts_planner.search(
                root_state=(current_gx, current_gy),
                goal_state=(sub_goal_gx, sub_goal_gy)
            )
            planning_time = time.time() - start_time
            
            with self.path_lock:
                if path_chunk and path_chunk[-1] == (sub_goal_gx, sub_goal_gy):
                    self.get_logger().info(f"Found path chunk with {len(path_chunk)} points in {planning_time:.2f}s.")
                    self.path_buffer = path_chunk
                    self.execution_index = 0
                    self.publish_path(self.path_buffer)
                else:
                    self.get_logger().warn(f"MCTS failed to find a complete path chunk after {planning_time:.2f}s.")
                    self.path_buffer = []
        
        except Exception as e:
            self.get_logger().error(f"Error in planning thread: {str(e)}\n{traceback.format_exc()}")
        finally:
            self.planner_active_flag = False

    def plan_next_path_chunk(self):
        if self.mcts_planner is None:
            self.get_logger().warn("Map not received, cannot plan.")
            return

        if self.planner_active_flag:
            self.get_logger().debug("Planning thread already active. Skipping new request.")
            return
            
        self.planner_thread = Thread(target=self._plan_next_path_chunk_async)
        self.planner_thread.start()

    def plan_and_move_callback(self):
        try:
            with self.path_lock:
                is_path_empty = not self.path_buffer
                path_length = len(self.path_buffer)
                current_goal_is_none = self.current_goal is None

            if is_path_empty or self.robot_current_pose is None or current_goal_is_none:
                self.publish_twist(0.0, 0.0)
                return
            
            current_wx = self.robot_current_pose.position.x
            current_wy = self.robot_current_pose.position.y
            goal_wx, goal_wy = self.current_goal
            
            distance_to_final_goal = math.hypot(goal_wx - current_wx, goal_wy - current_wy)
            goal_tolerance = self.get_parameter('goal_tolerance').value
            
            if distance_to_final_goal < goal_tolerance:
                self.get_logger().info("Final goal reached!")
                with self.path_lock:
                    self.path_buffer = []
                    self.current_goal = None
                self.publish_twist(0.0, 0.0)
                return
            
            # --- Re-planning trigger logic ---
            with self.path_lock:
                if self.execution_index >= (path_length - 5) and self.planner_active_flag is False:
                    self.plan_next_path_chunk()
            
            with self.path_lock:
                if not self.path_buffer:
                    self.publish_twist(0.0, 0.0)
                    return
                world_path = [self.grid_to_world(gx, gy) for gx, gy in self.path_buffer]

            min_dist = float('inf')
            closest_idx = self.execution_index
            for i in range(self.execution_index, len(world_path)):
                dist = math.hypot(world_path[i][0] - current_wx, world_path[i][1] - current_wy)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
            self.execution_index = closest_idx

            look_ahead_dist = self.get_parameter('look_ahead_distance').value
            target_idx = closest_idx
            while target_idx < len(world_path) - 1 and \
                    math.hypot(world_path[target_idx][0] - current_wx, world_path[target_idx][1] - current_wy) < look_ahead_dist:
                target_idx += 1
            
            self.current_lookahead_point = world_path[target_idx]
            
            target_wx, target_wy = self.current_lookahead_point
            dx = target_wx - current_wx
            dy = target_wy - current_wy
            target_angle = math.atan2(dy, dx)
            heading_error = (target_angle - self.current_theta + math.pi) % (2 * math.pi) - math.pi

            path_p1 = world_path[closest_idx]
            path_p2 = world_path[min(closest_idx + 1, len(world_path) - 1)]
            
            path_segment_dx = path_p2[0] - path_p1[0]
            path_segment_dy = path_p2[1] - path_p1[1]

            cte_numerator = (path_segment_dx * (path_p1[1] - current_wy) - 
                             (path_p1[0] - current_wx) * path_segment_dy)
            path_segment_length = math.hypot(path_segment_dx, path_segment_dy)
            cte = cte_numerator / (path_segment_length + 1e-6)

            angular_vel_gain = self.get_parameter('angular_vel_gain').value
            cte_gain = self.get_parameter('cte_gain').value
            angular_vel = (angular_vel_gain * heading_error) + (cte_gain * cte)

            max_linear_vel = self.get_parameter('max_linear_vel').value
            angle_based_scaling = max(0.1, 1.0 - 0.8 * abs(heading_error))
            linear_vel = max_linear_vel * angle_based_scaling

            max_angular_vel = self.get_parameter('max_angular_vel').value
            angular_vel = np.clip(angular_vel, -max_angular_vel, max_angular_vel)
            linear_vel = np.clip(linear_vel, self.get_parameter('min_linear_vel').value, max_linear_vel)

            self.publish_twist(linear_vel, angular_vel)
            self.publish_debug_markers()
            
        except Exception as e:
            self.get_logger().error(f"Error in plan_and_move_callback: {str(e)}\n{traceback.format_exc()}")
            self.publish_twist(0.0, 0.0)

    def world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        gx = int(round((wx - self.map_origin[0]) / self.map_resolution))
        gy = int(round((wy - self.map_origin[1]) / self.map_resolution))
        if self.mcts_planner:
            gx = max(0, min(gx, self.mcts_planner.grid_width - 1))
            gy = max(0, min(gy, self.mcts_planner.grid_height - 1))
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        wx = self.map_origin[0] + (gx + 0.5) * self.map_resolution
        wy = self.map_origin[1] + (gy + 0.5) * self.map_resolution
        return wx, wy

    def publish_path(self, path: List[Tuple[int, int]]):
        path_msg = Path()
        path_msg.header = Header(stamp=self.get_clock().now().to_msg(), frame_id='map')
        world_points = [self.grid_to_world(gx, gy) for gx, gy in path]
        path_msg.poses = [PoseStamped(header=path_msg.header, pose=Pose(position=Point(x=p[0], y=p[1], z=0.0))) for p in world_points]
        self.path_pub.publish(path_msg)

    def publish_twist(self, linear_x: float, angular_z: float):
        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        self.cmd_vel_pub.publish(twist)

    def publish_debug_markers(self):
        marker_array = MarkerArray()
        
        if self.current_lookahead_point:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "debug"
            marker.id = 1
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x, marker.pose.position.y = self.current_lookahead_point
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0
            marker.color.g = 1.0
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
        node.get_logger().error(f"Unhandled exception in main: {e}\n{traceback.format_exc()}")
    finally:
        node.get_logger().info("Shutting down MCTS planner node.")
        node.publish_twist(0.0, 0.0)
        
        # Wait for the planning thread to finish before destroying the node
        if node.planner_thread and node.planner_thread.is_alive():
            node.planner_thread.join()
            
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()