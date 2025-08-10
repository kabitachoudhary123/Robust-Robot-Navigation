# Robust-Robot-Navigation
Quantum-Guided Monte Carlo Tree Search for Robust Robot Navigation
TEAM:Q-Nav
MEMBER: KABITA CHOUDHARY

WISER Quantum Project 2025
1. Project Summary
Effective and efficient path planning in complex, obstacle-rich environments remains a fundamental challenge in mobile robotics. While traditional planners like A* guarantee optimality on a grid, they often produce kinematically challenging paths with many sharp turns. Conversely, sampling-based methods like Monte Carlo Tree Search (MCTS) can generate smoother paths but often suffer from slow convergence and unreliability on long-distance tasks. This paper introduces a novel hybrid planning architecture that synergizes the strengths of both approaches to produce high-quality, reliable paths for a mobile robot in a simulated warehouse. We propose an A-Guided Quantum MCTS Planner* that uses a fast A* search to generate a globally-aware heuristic "guideline." This guideline is then used to intelligently focus the simulation phase of a robust MCTS, which explores the solution space around the optimal path to discover safer and kinematically superior routes. The planner's performance is further enhanced with an adaptive iteration budget that scales with goal distance, ensuring a balance between solution quality and responsiveness. A key novelty of this work is the integration of a quantum search primitive to accelerate local decision-making. The MCTS selection phase is periodically enhanced with a simulated Grover's quantum search operator, which is shown in isolated benchmarks to identify the most promising branches with 100% accuracy, a significant improvement over a classical random guess (26% accuracy).
The complete system was implemented as a ROS 2 node and benchmarked against the standard Nav2 planner. The results are definitive: our planner consistently finds paths that are an order of magnitude smoother than the Nav2 default, with average smoothness values often below 1.5 rad compared to Nav2's 10-15 rad. While being computationally more intensive, our planner achieved a 100% success rate on a suite of challenging long-distance goals, establishing it as a robust and high-performance solution for generating superior trajectories in cluttered spaces.

2. System Overview & Key Features
This repository contains the source code for a complete, standalone ROS 2 navigation planner.

Key Features & Novelty
*Hybrid A-Guided MCTS:** Uses A* to provide a global heuristic, dramatically improving the efficiency and reliability of the MCTS search. The final path is the result of the MCTS, not the A*.

Quantum-Enhanced Selection: Integrates a simulated Grover's search operator as a proof-of-concept for accelerating decision-making at junctions.

Adaptive Planning Budget: Dynamically scales the number of MCTS iterations based on goal distance and a time-based cutoff.

Complete ROS 2 Integration: Includes configurable obstacle inflation for safety, a line-of-sight path smoother, and a robust waypoint controller.

Comprehensive Benchmarking: The project includes a standalone script to benchmark the performance of the Grover's search algorithm in isolation.

3. Results & Analysis
Path Planning Benchmark (vs. Nav2 Default)
The planner was tested on a suite of 7 challenging long-distance goals in a simulated warehouse.

Metric              Avg. Path Smoothness
Nav2 Default (A*)   10.95 rad
Our QMCTS Planner   0.84 rad ( >10x Smoother)

Conclusion: Our planner is significantly slower but produces paths of vastly superior quality, while maintaining perfect reliability on the test set.

Grover's Search Benchmark
The quantum selection component was tested in isolation on a mock problem of finding 2 correct items out of 8.

Method                  Success Rate
Classical Random Guess  26.0%
Quantum Search (Ideal)  100.0%
Quantum Search (Noisy)   100.0%


Conclusion: The ideal Grover's search provides a definitive advantage over classical guessing. The simulated noise from FakeManilaV2 was not sufficient to degrade the performance for this small-scale problem.

4. Installation & Setup
Prerequisites
Ubuntu 22.04 with ROS 2 Humble
Gazebo Simulator
Python 3.10+
The colcon build tool

Step 1: Install Python Dependencies
pip install --upgrade numpy matplotlib scipy qiskit qiskit-aer qiskit-ibm-runtime

Step 2: Build the ROS 2 Workspace
Clone this repository into the src folder of your ROS 2 workspace (e.g., ~/quantum/src/).

Navigate to the root of your workspace (e.g., ~/quantum/).

Clean and build the package:

rm -rf build install log
colcon build --packages-select montecarlowwalk
source install/setup.bash

5. How to Run
Running the Main Path Planner
Terminal 1: Launch the Simulation
Launch the Gazebo world, Nav2 components, RViz, and your custom planner node with a single command:

ros2 launch montecarlowwalk quantum_mcts_launch.py

Interact with RViz:
Use the "2D Pose Estimate" button to initialize the robot's position.
Use the "Nav2 Goal" button to set a destination. The robot will plan a path using the QMCTS planner and begin navigation.
Running the Grover's Search Benchmark
This is a standalone script and does not require Gazebo or RViz.
Build and source your workspace as described above.
Run the benchmark script from your workspace root:
ros2 run montecarlowwalk grover_benchmark
This will print the final comparison table to the terminal.
