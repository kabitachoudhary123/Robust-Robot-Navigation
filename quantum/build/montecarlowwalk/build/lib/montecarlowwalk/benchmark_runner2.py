import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose
import time
import math
import csv
from typing import List

class BenchmarkRunner(Node):
    def __init__(self):
        super().__init__('benchmark_runner')
        
        self.test_goals = [
            {'x': -6.73, 'y': -1.68, 'yaw': 1.29},
            {'x': 9.04,  'y': -4.26, 'yaw': 0.29},
            {'x': 9.49,  'y': -1.93, 'yaw': -0.08},
            {'x': -8.81, 'y': -3.39, 'yaw': 3.14},
            {'x': -9.52, 'y': 1.43,  'yaw': 0.10},
            {'x': -3.53, 'y': -1.62, 'yaw': 1.43},
            {'x': 9.88,  'y': 2.42,  'yaw': -1.57}
        ]
        self.current_goal_index = 0
        self.results = []
        self.state = 'IDLE' # Can be IDLE, TESTING_NAV2, TESTING_QMCTS

        self._qmcts_action_client = ActionClient(self, ComputePathToPose, 'compute_path_to_pose_qmcts')
        self._nav2_action_client = ActionClient(self, ComputePathToPose, 'compute_path_to_pose')

        self.get_logger().info("Benchmark Runner node started. Waiting for services...")
        self.timer = self.create_timer(15.0, self.start_benchmark)

    def start_benchmark(self):
        self.timer.cancel() # Run only once
        self.get_logger().info("Starting benchmark run...")
        self.run_next_goal()

    def run_next_goal(self):
        if self.current_goal_index >= len(self.test_goals):
            self.get_logger().info("All benchmarks complete. Writing results to file.")
            self.write_results_to_csv()
            rclpy.shutdown()
            return

        goal_data = self.test_goals[self.current_goal_index]
        self.get_logger().info(f"\n--- Running Benchmark for Goal {self.current_goal_index + 1}/{len(self.test_goals)}: ({goal_data['x']}, {goal_data['y']}) ---")
        
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = float(goal_data['x'])
        goal_pose.pose.position.y = float(goal_data['y'])
        goal_pose.pose.orientation.z = math.sin(goal_data['yaw'] / 2.0)
        goal_pose.pose.orientation.w = math.cos(goal_data['yaw'] / 2.0)
        
        self.current_goal_pose = goal_pose
        self.state = 'TESTING_NAV2'
        self.test_planner(self._nav2_action_client, self.nav2_goal_response_callback)

    def test_planner(self, action_client, response_callback):
        # --- THIS LINE IS CORRECTED ---
        self.get_logger().info(f"Testing planner for action server: '{action_client._action_name}'")
        if not action_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error(f"Action server '{action_client._action_name}' not available. Skipping.")
            # Record failure and move on
            if self.state == 'TESTING_NAV2':
                self.nav2_metrics = {'success': False, 'time': -1.0, 'length': -1.0, 'smoothness': -1.0}
                self.state = 'TESTING_QMCTS'
                self.test_planner(self._qmcts_action_client, self.qmcts_goal_response_callback)
            else: # Was testing QMCTS
                self.qmcts_metrics = {'success': False, 'time': -1.0, 'length': -1.0, 'smoothness': -1.0}
                self.record_and_advance()
            return

        goal_msg = ComputePathToPose.Goal()
        goal_msg.goal = self.current_goal_pose
        goal_msg.use_start = False

        self.send_goal_future = action_client.send_goal_async(goal_msg)
        self.send_goal_future.add_done_callback(response_callback)
        self.start_time = time.time()

    def nav2_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Nav2 planner rejected goal.')
            self.nav2_metrics = {'success': False, 'time': time.time() - self.start_time, 'length': 0.0, 'smoothness': 0.0}
            # Move on to testing QMCTS
            self.state = 'TESTING_QMCTS'
            self.test_planner(self._qmcts_action_client, self.qmcts_goal_response_callback)
            return

        self.get_result_future = goal_handle.get_result_async()
        self.get_result_future.add_done_callback(self.nav2_get_result_callback)
    
    def qmcts_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('QMCTS planner rejected goal.')
            self.qmcts_metrics = {'success': False, 'time': time.time() - self.start_time, 'length': 0.0, 'smoothness': 0.0}
            self.record_and_advance()
            return

        self.get_result_future = goal_handle.get_result_async()
        self.get_result_future.add_done_callback(self.qmcts_get_result_callback)

    def nav2_get_result_callback(self, future):
        result = future.result().result
        planning_time = time.time() - self.start_time
        self.nav2_metrics = self.calculate_metrics(result, planning_time)
        
        # Now test the next planner
        self.state = 'TESTING_QMCTS'
        self.test_planner(self._qmcts_action_client, self.qmcts_goal_response_callback)

    def qmcts_get_result_callback(self, future):
        result = future.result().result
        planning_time = time.time() - self.start_time
        self.qmcts_metrics = self.calculate_metrics(result, planning_time)
        self.record_and_advance()

    def record_and_advance(self):
        goal_data = self.test_goals[self.current_goal_index]
        self.results.append({
            'goal_x': goal_data['x'], 'goal_y': goal_data['y'],
            'nav2_success': self.nav2_metrics['success'], 'nav2_time': self.nav2_metrics['time'],
            'nav2_length': self.nav2_metrics['length'], 'nav2_smoothness': self.nav2_metrics['smoothness'],
            'qmcts_success': self.qmcts_metrics['success'], 'qmcts_time': self.qmcts_metrics['time'],
            'qmcts_length': self.qmcts_metrics['length'], 'qmcts_smoothness': self.qmcts_metrics['smoothness'],
        })
        self.current_goal_index += 1
        self.run_next_goal()

    def calculate_metrics(self, result, planning_time):
        metrics = {'success': False, 'time': planning_time, 'length': 0.0, 'smoothness': 0.0}
        if result and result.path.poses:
            metrics['success'] = True
            metrics['length'] = self.calculate_path_length(result.path.poses)
            metrics['smoothness'] = self.calculate_path_smoothness(result.path.poses)
        return metrics

    def calculate_path_length(self, poses: List[PoseStamped]) -> float:
        if len(poses) < 2: return 0.0
        length = 0.0
        for i in range(len(poses) - 1):
            p1 = poses[i].pose.position
            p2 = poses[i+1].pose.position
            length += math.hypot(p2.x - p1.x, p2.y - p1.y)
        return length

    def calculate_path_smoothness(self, poses: List[PoseStamped]) -> float:
        if len(poses) < 3: return 0.0
        smoothness = 0.0
        for i in range(1, len(poses) - 1):
            p_prev = poses[i-1].pose.position
            p_curr = poses[i].pose.position
            p_next = poses[i+1].pose.position
            angle1 = math.atan2(p_curr.y - p_prev.y, p_curr.x - p_prev.x)
            angle2 = math.atan2(p_next.y - p_curr.y, p_next.x - p_curr.x)
            angle_diff = (angle2 - angle1 + math.pi) % (2 * math.pi) - math.pi
            smoothness += abs(angle_diff)
        return smoothness

    def write_results_to_csv(self):
        if not self.results:
            self.get_logger().warn("No results to write.")
            return
        filename = f"benchmark_results_{int(time.time())}.csv"
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = self.results[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)
        self.get_logger().info(f"Benchmark results saved to {filename}")

def main(args=None):
    rclpy.init(args=args)
    node = BenchmarkRunner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Benchmark interrupted by user.")
    finally:
        node.write_results_to_csv()
        node.destroy_node()

if __name__ == '__main__':
    main()
