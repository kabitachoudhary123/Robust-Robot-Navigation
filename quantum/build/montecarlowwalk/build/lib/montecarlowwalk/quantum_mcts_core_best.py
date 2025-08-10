# quantum_mcts_core.py

import math
import numpy as np
import random
from typing import List, Tuple, Optional, Callable

# Make Qiskit optional: The planner will work without it, falling back to classical MCTS.
try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit.circuit.library import GroverOperator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

# --- Tuned MCTS Parameters ---
MAX_QUANTUM_STATES = 16
EXPLORATION_WEIGHT = 1.5 # Slightly increased to encourage exploring more options

class QuantumMCTSNode:
    """Represents a node in the Monte Carlo Search Tree."""
    def __init__(self, state: Tuple[int, int], parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = []

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def add_child(self, child_state: Tuple[int, int], untried_actions: List[Tuple[int, int]]):
        child_node = QuantumMCTSNode(child_state, parent=self)
        random.shuffle(untried_actions)
        child_node.untried_actions = untried_actions
        self.children.append(child_node)
        return child_node

    def best_child(self, exploration_weight=EXPLORATION_WEIGHT, use_quantum=False):
        if not self.children:
            return None

        if QISKIT_AVAILABLE and use_quantum and len(self.children) > 1 and len(self.children) <= MAX_QUANTUM_STATES:
            return self.quantum_best_child(exploration_weight)
        else:
            return self.classical_best_child(exploration_weight)

    def classical_best_child(self, exploration_weight: float):
        """Selects the best child using the classical UCT formula."""
        def uct(child):
            if child.visits == 0:
                return float('inf')
            exploitation_term = child.value / child.visits
            exploration_term = exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
            return exploitation_term + exploration_term
        
        return max(self.children, key=uct)

    def quantum_best_child(self, exploration_weight: float):
        """Selects a child using Grover's algorithm to amplify promising choices."""
        try:
            num_children = len(self.children)
            values = [(c.value / c.visits) if c.visits > 0 else 0.0 for c in self.children]
            threshold = np.percentile(values, 75) # Mark top 25% as "good"
            good_indices = [i for i, v in enumerate(values) if v >= threshold and v > 0]

            if not good_indices or len(good_indices) == num_children:
                return self.classical_best_child(exploration_weight)

            num_qubits = math.ceil(math.log2(num_children))
            target_states_bin = [format(i, f'0{num_qubits}b') for i in good_indices]
            
            grover = GroverOperator(target_states=target_states_bin)
            qc = QuantumCircuit(num_qubits)
            qc.h(range(num_qubits))
            qc.compose(grover, inplace=True)
            qc.measure_all()
            
            simulator = AerSimulator()
            result = simulator.run(qc, shots=10).result()
            counts = result.get_counts()
            
            selected_idx_str = max(counts.items(), key=lambda x: x[1])[0]
            selected_idx = int(selected_idx_str.split(' ')[-1], 2)
            
            return self.children[min(selected_idx, num_children - 1)]
        except Exception:
            return self.classical_best_child(exploration_weight)

class QuantumMCTSPlanner:
    """The core logic for the Quantum-enhanced Monte Carlo Tree Search planner."""
    def __init__(self, get_logger_func: Callable, grid_width: int, grid_height: int, obstacles: List[Tuple[int, int]]):
        self.get_logger = get_logger_func
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.obstacles = set(obstacles)
        
        self.max_iterations = 10000
        self.max_simulation_steps = 400
        self.simulation_epsilon = 0.2  # 20% chance of a random move during simulation

        self.goal_reward = 1000.0
        self.step_penalty = -1.0
        self.collision_penalty = -500.0
        self.distance_penalty_factor = 0.5
        self.simulation_cache = {}

    def is_valid_state(self, state: Tuple[int, int]) -> bool:
        x, y = state
        return (0 <= x < self.grid_width and 
                0 <= y < self.grid_height and 
                state not in self.obstacles)
    
    def heuristic_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def get_actions(self, state: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = state
        possible_actions = [
            (x+1, y), (x-1, y), (x, y+1), (x, y-1),
            (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)
        ]
        return [a for a in possible_actions if self.is_valid_state(a)]

    def simulate(self, start: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """
        Performs a simulation using an epsilon-greedy policy. This is vastly
        more effective than a purely random walk for pathfinding.
        """
        if (start, goal) in self.simulation_cache:
            return self.simulation_cache[(start, goal)]
        
        current = start
        total_reward = 0.0
        
        for _ in range(self.max_simulation_steps):
            if current == goal:
                total_reward += self.goal_reward
                break
            
            possible_next = self.get_actions(current)
            if not possible_next:
                total_reward += self.collision_penalty
                break
            
            # Epsilon-Greedy Policy
            if random.random() < self.simulation_epsilon:
                # Epsilon case: Take a random action
                next_state = random.choice(possible_next)
            else:
                # Greedy case: Take the best action (closest to the goal)
                next_state = min(possible_next, key=lambda s: self.heuristic_distance(s, goal))
            
            current = next_state
            total_reward += self.step_penalty
            total_reward -= self.heuristic_distance(current, goal) * self.distance_penalty_factor
        
        self.simulation_cache[(start, goal)] = total_reward
        return total_reward
    
    def search(self, root_state: Tuple[int, int], goal_state: Tuple[int, int]) -> List[Tuple[int, int]]:
        self.simulation_cache = {}
        root_node = QuantumMCTSNode(root_state)
        root_node.untried_actions = self.get_actions(root_state)
        
        for i in range(self.max_iterations):
            node = self.select(root_node)

            if node.state == goal_state:
                self.backpropagate(node, self.goal_reward)
                continue

            child = self.expand(node)
            if child:
                reward = self.simulate(child.state, goal_state)
                self.backpropagate(child, reward)
            else:
                self.backpropagate(node, self.collision_penalty)

            if i > 0 and i % 1000 == 0:
                self.get_logger().info(f"MCTS Iteration: {i}/{self.max_iterations}")

        path = []
        current = root_node
        while current and current.state != goal_state:
            path.append(current.state)
            if not current.children:
                self.get_logger().warn("MCTS search failed to find a complete path to the goal.")
                return []
            current = max(current.children, key=lambda c: c.visits)
        
        if current and current.state == goal_state:
            path.append(goal_state)
            self.get_logger().info(f"MCTS search complete. Final path has {len(path)} points.")
            return path
        
        self.get_logger().warn("MCTS finished without finding a viable path.")
        return []

    def select(self, node: QuantumMCTSNode) -> QuantumMCTSNode:
        while node.is_fully_expanded() and node.children:
            node = node.best_child(use_quantum=(node.visits % 10 == 0))
        return node
    
    def expand(self, node: QuantumMCTSNode) -> Optional[QuantumMCTSNode]:
        if not node.untried_actions:
            return None
        
        action = node.untried_actions.pop()
        if self.is_valid_state(action):
            return node.add_child(action, self.get_actions(action))
        return None

    def backpropagate(self, node: QuantumMCTSNode, reward: float):
        current = node
        while current:
            current.visits += 1
            current.value += reward
            current = current.parent