from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'montecarlowwalk'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

 
        # Include launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # Include world files
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),



        
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kabita',
    maintainer_email='kabita@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
             'mcts_planner_node = montecarlowwalk.mcts_planner_node:main',
             'quantum_mcts_node = montecarlowwalk.quantum_mcts_node:main',
             'benchmark_runner = montecarlowwalk.benchmark_runner:main', # Add this line
             'grover_benchmark = montecarlowwalk.grover_benchmark:run_grover_benchmark',
        ],
    },
)
