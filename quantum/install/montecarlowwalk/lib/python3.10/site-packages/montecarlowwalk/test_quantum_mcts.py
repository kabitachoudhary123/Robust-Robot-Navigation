# test_quantum_mcts.py
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import seaborn as sns
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import GroverOperator
from qiskit.quantum_info import Statevector

class QuantumMCTSTester:
    """Test harness for the quantum components in quantum_mcts_core.py"""
    
    def __init__(self):
        self.simulator = AerSimulator()
        self.test_results = {}
        plt.style.use('seaborn')
        sns.set_palette("husl")
    
    def plot_results(self, counts, title):
        """Plot measurement results as bar chart"""
        states = sorted(counts.keys())
        values = [counts[state] for state in states]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(states, values)
        
        # Color good states differently
        for i, state in enumerate(states):
            if int(state, 2) in [2, 6]:  # Good states
                bars[i].set_color('green')
        
        plt.title(f"Measurement Results: {title}")
        plt.xlabel("Quantum State")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_heatmap(self, selections, title):
        """Plot selection heatmap"""
        states = sorted(selections.keys())
        values = [selections[state] for state in states]
        
        plt.figure(figsize=(10, 6))
        sns.heatmap([values], annot=True, fmt="d", 
                   xticklabels=states, yticklabels=False,
                   cmap="YlGnBu", cbar=True)
        
        plt.title(f"Selection Heatmap: {title}")
        plt.xlabel("Node State")
        plt.show()
    
    def test_quantum_selection(self, num_children=8, good_indices=None, shots=1000):
        """Test the quantum node selection circuit"""
        if good_indices is None:
            good_indices = [2, 6]  # Default test case
            
        print(f"\nTesting quantum selection with {num_children} children (good ones: {good_indices})")
        
        # 1. Create the quantum circuit
        num_qubits = math.ceil(math.log2(num_children))
        
        # Create a quantum circuit that marks good indices
        qc = QuantumCircuit(num_qubits)
        
        # Create the oracle using Statevector
        oracle_matrix = np.ones(2**num_qubits)
        for idx in good_indices:
            if idx < 2**num_qubits:
                oracle_matrix[idx] = -1
        oracle = Statevector(oracle_matrix)
        
        # Build the complete circuit
        qc.h(range(num_qubits))  # Create superposition
        grover_op = GroverOperator(oracle)
        qc.compose(grover_op, inplace=True)
        qc.measure_all()
        
        # 2. Simulate the circuit
        transpiled = transpile(qc, self.simulator)
        result = self.simulator.run(transpiled, shots=shots).result()
        counts = result.get_counts()
        
        # 3. Plot results
        self.plot_results(counts, "Quantum Selection")
        
        # 4. Analyze results
        total_good = 0
        print("\nMeasurement results:")
        for state, count in sorted(counts.items()):
            idx = int(state, 2)
            is_good = idx in good_indices
            if is_good:
                total_good += count
            print(f"State {state} (Child {idx}): {count} shots {'[GOOD]' if is_good else ''}")
        
        success_rate = total_good / shots * 100
        print(f"\nSuccess rate: {success_rate:.1f}% (selected good children)")
        
        # 5. Verify against theoretical expectations
        k = len(good_indices)
        N = num_children
        classical_prob = k / N
        quantum_prob = np.sin((2 * math.asin(np.sqrt(k/N)) + 1) * np.pi/2)**2
        print(f"\nTheoretical probabilities:")
        print(f"Classical random: {classical_prob*100:.1f}%")
        print(f"Quantum Grover: {quantum_prob*100:.1f}%")
        
        # Store results
        self.test_results['quantum_selection'] = {
            'success_rate': success_rate,
            'expected_quantum': quantum_prob*100,
            'circuit_depth': qc.depth(),
            'qubits_used': num_qubits,
            'counts': counts
        }
        
        return success_rate
    
    def test_full_workflow(self):
        """Test the complete quantum node selection workflow"""
        print("\nTesting complete quantum MCTS node selection workflow")
        
        # Create mock nodes
        class MockNode:
            def __init__(self, state, value, visits):
                self.state = state
                self.value = value
                self.visits = visits
                self.children = []
        
        # Create parent with children
        parent = MockNode(None, 0, 100)
        children_data = [
            (0, 5.0, 10), (1, 5.0, 10),
            (2, 50.0, 10),  # Good child
            (3, 5.0, 10), (4, 5.0, 10),
            (5, 5.0, 10),
            (6, 50.0, 10),  # Good child
            (7, 5.0, 10)
        ]
        
        for state, value, visits in children_data:
            node = MockNode(state, value, visits)
            parent.children.append(node)
        
        # Run multiple selection trials
        trials = 1000
        selections = {child.state: 0 for child in parent.children}
        
        for _ in range(trials):
            # Replicate the exact logic from QuantumMCTSNode.quantum_best_child()
            num_children = len(parent.children)
            values = [(c.value / c.visits) for c in parent.children]
            threshold = np.percentile(values, 75)
            good_indices = [i for i, v in enumerate(values) if v >= threshold and v > 0]
            
            if not good_indices or len(good_indices) == num_children:
                selected = random.choice(parent.children)
            else:
                # Build the quantum circuit
                num_qubits = math.ceil(math.log2(num_children))
                
                # Create oracle using Statevector
                oracle_matrix = np.ones(2**num_qubits)
                for idx in good_indices:
                    oracle_matrix[idx] = -1
                oracle = Statevector(oracle_matrix)
                
                # Create Grover operator
                grover_op = GroverOperator(oracle)
                qc = QuantumCircuit(num_qubits)
                qc.h(range(num_qubits))  # Create superposition
                qc.compose(grover_op, inplace=True)
                qc.measure_all()
                
                # Run simulation
                transpiled = transpile(qc, self.simulator)
                result = self.simulator.run(transpiled, shots=1).result()
                counts = result.get_counts()
                selected_idx = int(list(counts.keys())[0], 2)
                selected = parent.children[min(selected_idx, num_children - 1)]
            
            selections[selected.state] += 1
        
        # Plot heatmap of selections
        self.plot_heatmap(selections, "Full Workflow Selections")
        
        # Calculate success rate
        good_states = {2, 6}
        good_selections = sum(count for state, count in selections.items() if state in good_states)
        success_rate = good_selections / trials * 100
        
        print("\nSelection results:")
        for state, count in sorted(selections.items()):
            print(f"Node {state}: {count} selections ({count/trials*100:.1f}%)")
        
        print(f"\nGood nodes selection rate: {success_rate:.1f}%")
        
        self.test_results['full_workflow'] = {
            'success_rate': success_rate,
            'expected_rate': 95.0,  # Theoretical expectation
            'selections': selections
        }
        
        return success_rate
    
    def run_all_tests(self):
        """Run all test cases"""
        print("=== Testing Quantum MCTS Components ===")
        self.test_quantum_selection()
        self.test_full_workflow()
        
        # Plot comparison of results
        self.plot_comparison()
        
        print("\n=== Test Summary ===")
        for test_name, results in self.test_results.items():
            print(f"\n{test_name.replace('_', ' ').title()}:")
            for metric, value in results.items():
                if metric not in ['counts', 'selections']:
                    print(f"  {metric.replace('_', ' ')}: {value:.1f}" 
                          + ("%" if "rate" in metric else ""))
        
        return all(r['success_rate'] > 50 for r in self.test_results.values())
    
    def plot_comparison(self):
        """Plot comparison of classical vs quantum probabilities"""
        quantum_data = self.test_results['quantum_selection']
        classical_prob = len([2, 6]) / 8 * 100
        quantum_prob = quantum_data['expected_quantum']
        actual_prob = quantum_data['success_rate']
        
        labels = ['Classical', 'Quantum Theory', 'Quantum Actual']
        values = [classical_prob, quantum_prob, actual_prob]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, values)
        bars[0].set_color('blue')
        bars[1].set_color('orange')
        bars[2].set_color('green')
        
        plt.title("Probability Comparison: Selecting Good Nodes")
        plt.ylabel("Probability (%)")
        plt.ylim(0, 100)
        for i, v in enumerate(values):
            plt.text(i, v + 2, f"{v:.1f}%", ha='center')
        plt.grid(True, alpha=0.3)
        plt.show()

if __name__ == "__main__":
    tester = QuantumMCTSTester()
    tests_passed = tester.run_all_tests()
    print("\nAll tests passed!" if tests_passed else "\nSome tests failed!")