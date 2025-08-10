import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch_ros.actions import Node, LifecycleNode
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Paths
    gazebo_ros_pkg = get_package_share_directory("gazebo_ros")
    montecarlowwalk_pkg = get_package_share_directory("montecarlowwalk")
    turtlebot3_pkg = get_package_share_directory("turtlebot3_gazebo")
    nav2_map_server_pkg = get_package_share_directory("nav2_map_server")
    nav2_lifecycle_manager_pkg = get_package_share_directory("nav2_lifecycle_manager")

    # World file
    world_file = PathJoinSubstitution([montecarlowwalk_pkg, "worlds", "warehouse.world"])

    # Robot model file
    urdf_file = PathJoinSubstitution([turtlebot3_pkg, "urdf", "turtlebot3_burger.urdf"])
    robot_description_content = Command(['xacro', ' ', urdf_file])

    # Map file
    map_file = "/home/kabita/ros2_ws/src/warehouse_sim/maps/map.yaml"

    # Gazebo launch
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_pkg, "launch", "gazebo.launch.py")
        ),
        launch_arguments={"world": world_file}.items(),
    )

    # Spawn the robot
    model_file = PathJoinSubstitution([turtlebot3_pkg, "models", "turtlebot3_burger", "model.sdf"])
    robot_spawn = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-entity", "turtlebot3_burger",
            "-file", model_file,
            "-x", "0", "-y", "0", "-z", "0.1"
        ],
        output="screen"
    )

    # RViz2 node
    rviz_config_file = PathJoinSubstitution([montecarlowwalk_pkg, "rviz", "warehouse.rviz"])
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config_file],
        parameters=[{"use_sim_time": True}]
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"use_sim_time": True, "robot_description": robot_description_content}],
    )

    # Joint state publisher
    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        output="screen",
        parameters=[{"use_sim_time": True}]
    )
    
    # Map server
    map_server = LifecycleNode(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        namespace="",
        output="screen",
        parameters=[
            {'use_sim_time': True},
            {'yaml_filename': map_file},
            {'topic_name': 'map'},
            {'frame_id': 'map'}
        ],
    )

    # AMCL node
    amcl_node = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            {'set_initial_pose': True},
            {'initial_pose.x': 0.0},
            {'initial_pose.y': 0.0},
            {'initial_pose.yaw': 0.0}
        ]
    )

    # Delayed lifecycle manager
    lifecycle_manager = TimerAction(
        period=5.0,
        actions=[
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_navigation",
                output="screen",
                parameters=[
                    {"autostart": True},
                    {"use_sim_time": True},
                    {"node_names": ["map_server", "amcl"]}
                ]
            )
        ]
    )

    # Quantum MCTS Planner Node with optimized parameters
    quantum_mcts_node = TimerAction(
    period=7.0,  # Start a little earlier
    actions=[
        Node(
            package="montecarlowwalk", # Your package name
            executable="quantum_mcts_node", # Your executable name
            name="quantum_mcts_node",
            output="screen",
            parameters=[{
            'use_sim_time': True,
            'iterations_per_meter': 2500,
            'min_iterations': 5000,
            'max_iterations': 60000,
            'max_planning_time': 15.0,
            'heuristic_weight': 15.0,
            'max_linear_vel': 0.22,
            'min_linear_vel': 0.05,
            'max_angular_vel': 1.0,
            'angular_vel_gain': 1.8,
            'turn_in_place_threshold': 0.7,
            'waypoint_tolerance': 0.2,
            'goal_tolerance': 0.15,
            'inflation_radius': 0.25
        }]
        )
    ]
)

    return LaunchDescription([
        DeclareLaunchArgument("world", default_value=world_file, description="Path to the warehouse world file"),
        gazebo_launch,
        robot_spawn,
        robot_state_publisher,
        joint_state_publisher,
        map_server,
        amcl_node,
        lifecycle_manager,
        rviz_node,
        quantum_mcts_node,  # Replaced mcts_planner_node with quantum version
    ])