# quantum_mcts_core.py
import math
import numpy as np
import random
import time
from typing import List, Tuple, Optional, Callable, Set

try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit.circuit.library import GroverOperator
    from qiskit.quantum_info import PhaseOracle
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

MAX_QUANTUM_STATES = 16
EXPLORATION_WEIGHT = 2.0

class QuantumMCTSNode:
    def __init__(self, state: Tuple[int, int], parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = []
        self.get_logger = lambda: None 

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def add_child(self, child_state: Tuple[int, int], untried_actions: List[Tuple[int, int]]):
        child_node = QuantumMCTSNode(child_state, parent=self)
        random.shuffle(untried_actions)
        child_node.untried_actions = untried_actions
        child_node.get_logger = self.get_logger
        self.children.append(child_node)
        return child_node

    def best_child(self, goal_state: Tuple[int, int], heuristic_fn: Callable, heuristic_weight: float, use_quantum=False):
        if not self.children: return None
        if QISKIT_AVAILABLE and use_quantum and len(self.children) > 1 and len(self.children) <= MAX_QUANTUM_STATES:
            return self.quantum_best_child(goal_state, heuristic_fn, heuristic_weight)
        else:
            return self.classical_best_child(goal_state, heuristic_fn, heuristic_weight)

    def classical_best_child(self, goal_state: Tuple[int, int], heuristic_fn: Callable, heuristic_weight: float):
        best_score = -float('inf')
        best_children = []
        for child in self.children:
            if child.visits == 0: return child
            exploitation_term = child.value / child.visits
            exploration_term = EXPLORATION_WEIGHT * math.sqrt(math.log(self.visits) / child.visits)
            dist_to_goal = heuristic_fn(child.state, goal_state)
            heuristic_term = heuristic_weight / (1.0 + dist_to_goal)
            score = exploitation_term + exploration_term + heuristic_term
            if score > best_score:
                best_score = score
                best_children = [child]
            elif score == best_score:
                best_children.append(child)
        return random.choice(best_children)

    def quantum_best_child(self, goal_state: Tuple[int, int], heuristic_fn: Callable, heuristic_weight: float):
        try:
            num_children = len(self.children)
            values = [(c.value / c.visits) if c.visits > 0 else 0.0 for c in self.children]
            threshold = np.percentile(values, 75)
            good_indices = [i for i, v in enumerate(values) if v >= threshold and v > 0]
            if not good_indices or len(good_indices) == num_children:
                return self.classical_best_child(goal_state, heuristic_fn, heuristic_weight)
            num_qubits = math.ceil(math.log2(num_children))
            target_states_bin = [format(i, f'0{num_qubits}b') for i in good_indices]
            oracle = PhaseOracle(target_states_bin)
            grover_op = GroverOperator(oracle)
            qc = QuantumCircuit(grover_op.num_qubits)
            qc.h(range(num_qubits))
            qc.compose(grover_op, inplace=True)
            qc.measure_all()
            simulator = AerSimulator()
            result = simulator.run(qc, shots=10).result()
            counts = result.get_counts()
            selected_idx = int(max(counts.items(), key=lambda x: x[1])[0].split(' ')[-1], 2)
            return self.children[min(selected_idx, num_children - 1)]
        except Exception as e:
            self.get_logger().warn(f"Quantum selection failed ({e}), falling back to classical method.")
            return self.classical_best_child(goal_state, heuristic_fn, heuristic_weight)

class QuantumMCTSPlanner:
    def __init__(self, get_logger_func: Callable, grid_width: int, grid_height: int, obstacles: Set[Tuple[int, int]], heuristic_weight: float, distance_penalty_factor: float):
        self.get_logger = get_logger_func
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.obstacles = obstacles
        self.heuristic_weight = heuristic_weight
        self.distance_penalty_factor = distance_penalty_factor # Now a parameter
        self.max_simulation_steps = 400
        self.simulation_epsilon = 0.2
        self.goal_reward = 1000.0
        self.step_penalty = -1.0
        self.collision_penalty = -500.0
        self.simulation_cache = {}

    def is_valid_state(self, state: Tuple[int, int]) -> bool:
        x, y = state
        return (0 <= x < self.grid_width and 0 <= y < self.grid_height and state not in self.obstacles)
    
    def heuristic_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def get_actions(self, state: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = state
        possible_actions = [(x+1, y), (x-1, y), (x, y+1), (x, y-1), (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)]
        return [a for a in possible_actions if self.is_valid_state(a)]

    # --- REVERTED to the more powerful, intelligent simulation logic ---
    def simulate(self, start: Tuple[int, int], goal: Tuple[int, int]) -> float:
        if (start, goal) in self.simulation_cache: return self.simulation_cache[(start, goal)]
        
        current, total_reward = start, 0.0
        for _ in range(self.max_simulation_steps):
            if current == goal:
                total_reward += self.goal_reward
                break
            
            possible_next = self.get_actions(current)
            if not possible_next:
                total_reward += self.collision_penalty
                break
            
            current = random.choice(possible_next) if random.random() < self.simulation_epsilon else min(possible_next, key=lambda s: self.heuristic_distance(s, goal))
            
            # This per-step penalty provides a rich gradient for the search
            total_reward += self.step_penalty
            total_reward -= self.heuristic_distance(current, goal) * self.distance_penalty_factor
        
        self.simulation_cache[(start, goal)] = total_reward
        return total_reward
    
    def search(self, root_state: Tuple[int, int], goal_state: Tuple[int, int], max_iterations: int, max_planning_time: float) -> List[Tuple[int, int]]:
        self.simulation_cache = {}
        root_node = QuantumMCTSNode(root_state)
        root_node.untried_actions = self.get_actions(root_state)
        root_node.get_logger = self.get_logger
        start_time = time.time()
        for i in range(max_iterations):
            if time.time() - start_time > max_planning_time:
                self.get_logger().warn(f"Planning time limit ({max_planning_time}s) reached after {i} iterations.")
                break
            node = self.select(root_node, goal_state)
            if node.state == goal_state:
                self.backpropagate(node, self.goal_reward)
                continue
            child = self.expand(node)
            if child:
                reward = self.simulate(child.state, goal_state)
                self.backpropagate(child, reward)
            else:
                self.backpropagate(node, self.collision_penalty)
        path, current = [], root_node
        while current and current.state != goal_state:
            path.append(current.state)
            if not current.children: return []
            current = max(current.children, key=lambda c: c.visits)
        if current and current.state == goal_state:
            path.append(goal_state)
            return path
        return []

    def select(self, node: QuantumMCTSNode, goal_state: Tuple[int, int]) -> QuantumMCTSNode:
        while node.is_fully_expanded() and node.children:
            node = node.best_child(goal_state, self.heuristic_distance, self.heuristic_weight, use_quantum=(node.visits % 10 == 0))
        return node
    
    def expand(self, node: QuantumMCTSNode) -> Optional[QuantumMCTSNode]:
        if not node.untried_actions: return None
        action = node.untried_actions.pop()
        return node.add_child(action, self.get_actions(action)) if self.is_valid_state(action) else None

    def backpropagate(self, node: QuantumMCTSNode, reward: float):
        current = node
        while current:
            current.visits += 1
            current.value += reward
            current = current.parent