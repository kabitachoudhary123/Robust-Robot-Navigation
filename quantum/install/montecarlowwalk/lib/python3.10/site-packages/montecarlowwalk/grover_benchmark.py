import numpy as np
import random
import time
import math
from typing import List, Tuple, Optional, Callable

# --- Qiskit Imports ---
# All necessary Qiskit components are included here.

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import GroverOperator
from qiskit.circuit.library import PhaseOracle
# --- FINAL, CORRECTED IMPORT for modern Qiskit versions ---
from qiskit_ibm_runtime.fake_provider import FakeManilaV2 as FakeBackend
QISKIT_AVAILABLE = True
print("Found modern Qiskit V2 FakeBackend (FakeManilaV2).")


# --- Self-Contained MCTS Node Class ---
# The class containing your quantum circuit is copied directly into this file.

MAX_QUANTUM_STATES = 16
EXPLORATION_WEIGHT = 2.0

class QuantumMCTSNode:
    def __init__(self, state: Tuple[int, int], parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        #self.get_logger = lambda: None 

    def classical_best_child(self, heuristic_fn, heuristic_weight, goal_state=None):
        best_score = -float('inf')
        best_children = []
        for child in self.children:
            if child.visits == 0: return child
            exploitation_term = child.value / child.visits
            exploration_term = EXPLORATION_WEIGHT * math.sqrt(math.log(self.visits) / child.visits)
            dist_to_goal = heuristic_fn(child.state, goal_state) if heuristic_fn and goal_state else 0
            heuristic_term = heuristic_weight / (1.0 + dist_to_goal)
            score = exploitation_term + exploration_term + heuristic_term
            if score > best_score:
                best_score = score
                best_children = [child]
            elif score == best_score:
                best_children.append(child)
        return random.choice(best_children)

    def quantum_best_child(self, heuristic_fn, heuristic_weight, goal_state=None, simulator=None):
        num_children = len(self.children)
        values = [(c.value / c.visits) if c.visits > 0 else 0.0 for c in self.children]
        threshold = np.percentile(values, 75)
        good_indices = [i for i, v in enumerate(values) if v >= threshold and v > 0]
        if not good_indices or len(good_indices) == num_children:
            return self.classical_best_child(heuristic_fn, heuristic_weight, goal_state)

        num_qubits = math.ceil(math.log2(num_children))

        def _bitstring_to_expression(bitstring):
            # --- FIX #1: Reverse the bitstring to match Qiskit's variable order ---
            bitstring = bitstring[::-1]
            terms = [f'~x_{i}' if bit == '0' else f'x_{i}' for i, bit in enumerate(bitstring)]
            return f"({' & '.join(terms)})"

        target_states_bin = [format(i, f'0{num_qubits}b') for i in good_indices]
        boolean_expression = ' | '.join([_bitstring_to_expression(bs) for bs in target_states_bin])
        
        oracle = PhaseOracle(boolean_expression)
        num_iterations = 1
        
        # --- FIX #2: Create a single operator, then use .power() for iterations ---
        single_iteration_op = GroverOperator(oracle)
        grover_op = single_iteration_op.power(num_iterations)

        # Build the full circuit
        qc = QuantumCircuit(grover_op.num_qubits)
        qc.h(range(num_qubits))
        qc.compose(grover_op, inplace=True)
        
        # --- FIX #3 (CRUCIAL): Decompose the high-level operator into basic gates ---
        qc = qc.decompose().decompose() 
        
        qc.measure_all()
        
        if simulator is None:
            simulator = AerSimulator()
        transpiled_qc = transpile(qc, backend=simulator)
        result = simulator.run(transpiled_qc, shots=10).result()
        #result = simulator.run(qc, shots=10).result()
        counts = result.get_counts()
        selected_idx = int(max(counts.items(), key=lambda x: x[1])[0].split(' ')[-1], 2)
        return self.children[min(selected_idx, num_children - 1)]
# --- Main Benchmark Function ---

def run_grover_benchmark():
    """
    Isolates and tests the performance of the quantum_best_child function.
    """
    if not QISKIT_AVAILABLE:
        print("Qiskit not available. Aborting benchmark.")
        return

    print("\n--- Benchmarking Grover's Search for MCTS Node Selection ---")

    # 1. Setup the Test Scenario
    num_children = 8
    good_indices = {2, 6} 
    num_trials = 100

    parent_node = QuantumMCTSNode(state=(0,0))
    parent_node.visits = 100

    class DummyLogger:
        def warn(self, msg):
            print(f"[DUMMY LOGGER-WARN]: {msg}")
    #parent_node.get_logger = lambda: DummyLogger()

    for i in range(num_children):
        child = QuantumMCTSNode(state=(i,i), parent=parent_node)
        child.visits = 10
        child.value = 50.0 if i in good_indices else 5.0
        parent_node.children.append(child)

    # 2. Run Classical Random Search (Baseline)
    print(f"\nRunning {num_trials} trials of Classical Random Search...")
    classical_successes = 0
    for _ in range(num_trials):
        choice = random.choice(parent_node.children)
        if choice.state[0] in good_indices:
            classical_successes += 1
    classical_accuracy = (classical_successes / num_trials) * 100
    print(f"Classical Accuracy: {classical_accuracy:.1f}%")

    # 3. Run Quantum Search (Noiseless)
    print(f"\nRunning {num_trials} trials of Noiseless Quantum Search...")
    quantum_noiseless_successes = 0
    for _ in range(num_trials):
        choice = parent_node.quantum_best_child(heuristic_fn=lambda a,b: 0, heuristic_weight=0)
        if choice.state[0] in good_indices:
            quantum_noiseless_successes += 1
    quantum_noiseless_accuracy = (quantum_noiseless_successes / num_trials) * 100
    print(f"Noiseless Quantum Accuracy: {quantum_noiseless_accuracy:.1f}%")

    # 4. Run Quantum Search (Noisy)
    if FakeBackend:
        print(f"\nRunning {num_trials} trials of Noisy Quantum Search...")
        noisy_backend = FakeBackend()
        noisy_simulator = AerSimulator.from_backend(noisy_backend)
        
        quantum_noisy_successes = 0
        for _ in range(num_trials):
            # Pass the noisy simulator directly as an argument
            choice = parent_node.quantum_best_child(heuristic_fn=lambda a,b: 0, heuristic_weight=0, simulator=noisy_simulator)
            if choice.state[0] in good_indices:
                quantum_noisy_successes += 1
        quantum_noisy_accuracy = (quantum_noisy_successes / num_trials) * 100
        print(f"Noisy Quantum Accuracy: {quantum_noisy_accuracy:.1f}%")
    else:
        quantum_noisy_accuracy = "N/A"

    # 5. Print Final Results
    print("\n--- Grover Search Benchmark Results ---")
    print(f"Scenario: Find {len(good_indices)} correct items out of {num_children}.")
    print("-" * 40)
    print(f"| Method                  | Success Rate |")
    print(f"|-------------------------|--------------|")
    print(f"| Classical Random Guess  | {classical_accuracy:^12.1f}% |")
    print(f"| Quantum Search (Ideal)  | {quantum_noiseless_accuracy:^12.1f}% |")
    if FakeBackend:
        print(f"| Quantum Search (Noisy)  | {quantum_noisy_accuracy:^12.1f}% |")
    print("-" * 40)

def main():
    """Entry point for the ROS 2 executable."""
    run_grover_benchmark()

if __name__ == '__main__':
    main()
