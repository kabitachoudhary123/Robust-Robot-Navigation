import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, expon
import heapq
import random
import math

# Import Qiskit
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

# --- ROBUST IMPORT for Fake Backend ---
try:
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2 as FakeBackend
    print("Found modern Qiskit V2 FakeBackend (FakeManilaV2).")
except ImportError:
    print("="*80)
    print("WARNING: Could not find the modern Qiskit V2 fake backend.")
    print("Please run 'pip install --upgrade qiskit qiskit-aer qiskit-ibm-runtime'")
    print("Noisy simulation tasks will be skipped.")
    print("="*80)
    FakeBackend = None

# We can reuse the core logic from our planner!
from montecarlowwalk.quantum_mcts_core import QuantumMCTSPlanner 

# --- Performance Metric Calculation (Task 5) ---

def get_distribution_from_counts(counts, num_qubits):
    """Converts Qiskit counts dictionary to a probability vector."""
    num_outcomes = 2**num_qubits
    distribution = np.zeros(num_outcomes)
    total_shots = sum(counts.values())
    
    for bitstring, count in counts.items():
        index = int(bitstring, 2)
        if index < num_outcomes:
            distribution[index] = count / total_shots
            
    return distribution

def kl_divergence(p, q):
    """Calculates KL divergence."""
    epsilon = 1e-12
    if len(p) > len(q):
        q = np.pad(q, (0, len(p) - len(q)), 'constant', constant_values=epsilon)
    elif len(q) > len(p):
        p = np.pad(p, (0, len(q) - len(p)), 'constant', constant_values=epsilon)
        
    p = np.asarray(p, dtype=np.float64) + epsilon
    q = np.asarray(q, dtype=np.float64) + epsilon
    p /= np.sum(p)
    q /= np.sum(q)
    return np.sum(p * np.log(p / q))

# --- Experiment Functions ---

def run_warehouse_walk(grid_width, grid_height, num_walks, num_steps, walk_type='gaussian'):
    """Runs Monte Carlo simulations in a simple open grid."""
    start_pos = (grid_width // 2, grid_height // 2)
    final_positions = []
    termination_prob = 0.2

    for _ in range(num_walks):
        current_pos = start_pos
        for step in range(num_steps):
            if walk_type == 'exponential' and random.random() < termination_prob:
                break

            x, y = current_pos
            move = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
            dx, dy = move
            new_x, new_y = x + dx, y + dy

            if 0 <= new_x < grid_width and 0 <= new_y < grid_height:
                current_pos = (new_x, new_y)
        
        final_positions.append(current_pos)
    
    return np.array(final_positions)

# --- THIS IS THE FINAL, OPTIMIZED QUANTUM WALK IMPLEMENTATION ---

def qft_rotations(circuit, register):
    """Manually constructs the rotation part of the QFT on a specific register."""
    n = len(register)
    if n == 0:
        return
    
    # Apply rotations starting from the most significant qubit
    for i in range(n):
        qubit = register[i]
        circuit.h(qubit)
        for j in range(i + 1, n):
            control_qubit = register[j]
            # The angle is pi / 2^(j-i)
            circuit.cp(np.pi / (2**(j - i)), control_qubit, qubit)

def swap_registers(circuit, register):
    """Applies the swap gates at the end of the QFT on a specific register."""
    n = len(register)
    for i in range(n // 2):
        circuit.swap(register[i], register[n - 1 - i])

def qft(circuit, register):
    """Applies a manual QFT to a specific register."""
    qft_rotations(circuit, register)
    swap_registers(circuit, register)

def iqft(circuit, register):
    """Applies a manual inverse QFT to a specific register."""
    n = len(register)
    # The inverse QFT is the dagger of the QFT circuit.
    # We can build it by reversing the operations.
    
    # First, apply the swaps
    swap_registers(circuit, register)
    # Then, apply the inverse of the rotations
    for i in reversed(range(n)):
        qubit = register[i]
        for j in reversed(range(i + 1, n)):
            control_qubit = register[j]
            circuit.cp(-np.pi / (2**(j - i)), control_qubit, qubit)
        circuit.h(qubit)


def run_manual_qft_quantum_walk(steps, position_qubits, shots=8192, use_noise=False):
    """Simulates a 1D Quantum Walk using a manually implemented QFT."""
    
    num_qubits = position_qubits + 1
    qc = QuantumCircuit(num_qubits, position_qubits)
    
    position_register = list(range(position_qubits))
    coin_qubit = position_qubits

    def controlled_adder(qc, sign):
        """Builds the QFT-based adder/subtractor."""
        # 1. Apply QFT to the position register
        qft(qc, position_register)
        
        # 2. Apply controlled phase additions
        for i in range(position_qubits):
            angle = sign * np.pi / (2**i)
            # The control is the coin, the target is the position qubit
            qc.cp(angle, coin_qubit, position_register[position_qubits - 1 - i])
            
        # 3. Apply inverse QFT
        iqft(qc, position_register)

    # 1. Initialize state (start at 0 for simplicity)
    qc.h(coin_qubit)
    qc.barrier()

    # 2. Main walk loop
    for _ in range(steps):
        qc.h(coin_qubit) # Coin flip
        
        controlled_adder(qc, 1.0)  # Increment if coin is |1>
        qc.x(coin_qubit)
        controlled_adder(qc, -1.0) # Decrement if coin is |0>
        qc.x(coin_qubit)
        
        qc.barrier()

    # 3. Measure
    qc.measure(position_register, position_register)

    # 4. Simulate
    if use_noise:
        if FakeBackend is None: return None, None
        backend = FakeBackend()
        simulator = AerSimulator.from_backend(backend)
    else:
        simulator = AerSimulator()
        
    # Use the highest optimization level for the best result on noisy hardware
    transpiled_qc = transpile(qc, simulator, optimization_level=3)
    result = simulator.run(transpiled_qc, shots=shots).result()
    counts = result.get_counts(0)
    
    return counts, transpiled_qc

# --- Main Analysis Function ---

def main_analysis():
    GRID_WIDTH, GRID_HEIGHT = 101, 101
    NUM_WALKS = 5000
    NUM_STEPS = 30

    # --- Task 2: Gaussian Distribution ---
    print("\n--- Task 2: Simulating Gaussian-like Distribution ---")
    final_pos_gaussian = run_warehouse_walk(GRID_WIDTH, GRID_HEIGHT, NUM_WALKS, NUM_STEPS, 'gaussian')
    plt.figure(figsize=(8, 8))
    plt.hist2d(final_pos_gaussian[:, 0], final_pos_gaussian[:, 1], bins=30, cmap='viridis')
    plt.colorbar(label='Number of Walks Ended in Cell')
    plt.title(f'Distribution of {NUM_WALKS} Random Walks ({NUM_STEPS} steps)')
    plt.xlabel('Final X Position')
    plt.ylabel('Final Y Position')
    plt.show()

    # --- Task 3a: Exponential Distribution ---
    print("\n--- Task 3a: Simulating Exponential-like Distribution ---")
    final_pos_exp = run_warehouse_walk(GRID_WIDTH, GRID_HEIGHT, NUM_WALKS, NUM_STEPS, 'exponential')
    start_pos_exp = (GRID_WIDTH // 2, GRID_HEIGHT // 2)
    distances = [math.hypot(pos[0] - start_pos_exp[0], pos[1] - start_pos_exp[1]) for pos in final_pos_exp]
    plt.figure(figsize=(8, 6))
    plt.hist(distances, bins=50, density=True, label='Simulated Distribution', alpha=0.7)
    x = np.linspace(0, max(distances), 100)
    p = expon.fit(distances)
    r = expon.pdf(x, *p)
    plt.plot(x, r, 'r--', linewidth=2, label='Fitted Exponential Curve')
    plt.title('Distribution of Distances from Start (Exponential Walk)')
    plt.xlabel('Distance from Start')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    # --- Task 3b & 4: Optimized Quantum Walk ---
    print("\n--- Task 3b & 4: Optimized Quantum Walk ---")
    if FakeBackend is not None:
        POSITION_QUBITS = 4
        QW_STEPS = 4
        
        counts_ideal, _ = run_manual_qft_quantum_walk(QW_STEPS, POSITION_QUBITS, use_noise=False)
        dist_ideal = get_distribution_from_counts(counts_ideal, POSITION_QUBITS)
        
        counts_noisy, qc_noisy_transpiled = run_manual_qft_quantum_walk(QW_STEPS, POSITION_QUBITS, use_noise=True)
        if counts_noisy:
            dist_noisy = get_distribution_from_counts(counts_noisy, POSITION_QUBITS)
            
            print(f"Noisy Circuit Depth: {qc_noisy_transpiled.depth()}")
            print(f"Noisy Circuit CNOT count: {qc_noisy_transpiled.count_ops().get('cx', 0)}")
            
            print("\n--- Task 5: Performance Metrics ---")
            kl_noisy_vs_ideal = kl_divergence(dist_ideal, dist_noisy)
            print(f"KL Divergence (Noisy vs. Ideal): {kl_noisy_vs_ideal:.6f}")

            plot_histogram([counts_ideal, counts_noisy], legend=['Ideal', 'Noisy'], title='Optimized QFT-Based Quantum Walk')
            plt.show()
    else:
        print("Skipping noisy simulation tasks as a fake backend was not found.")


if __name__ == '__main__':
    main_analysis()
