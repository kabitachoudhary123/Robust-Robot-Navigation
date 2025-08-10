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

    # MCTS Planner Node
    mcts_planner_node = Node(
        package="montecarlowwalk",
        executable="mcts_planner_node",
        name="mcts_planner_node",
        output="screen",
        parameters=[
            {"use_sim_time": True},
            {"timer_period": 0.05},          # Very fast control loop (20Hz)
            {"max_linear_vel": 0.08},         # Reduced maximum speed
            {"min_linear_vel": 0.02},        # Minimum speed
            {"angular_vel_gain": 0.8},       # Balanced rotation control
            {"goal_tolerance": 0.05},        # Very tight goal tolerance
            {"waypoint_tolerance": 0.08},    # Tight waypoint tolerance
            {"max_iterations": 75000},        # Faster planning
            {"max_simulation_steps": 200},    # Faster planning
            {"exploration_weight": 1.0},      # More exploitation
            {"replan_threshold": 0.2},        # More sensitive to deviations
            {"smoothing_factor": 0.7},       # Balanced smoothing
            {"look_ahead_distance": 0.3},     # Short lookahead
            {"min_look_ahead": 0.15},        # Minimum lookahead
            {"max_look_ahead": 0.5},         # Limited maximum lookahead
            {"path_follow_gain": 1.2},       # More aggressive path following
            {"braking_distance": 0.5},       # Start braking earlier
            {"stop_distance": 0.08},        # Final stopping distance
            {"max_deceleration": 0.2},      # Gentle braking
            {"path_update_interval": 0.5},   # Frequent path updates
            {"use_speed_adaptation": True},  # Enable speed adaptation
            {"speed_adaptation_factor": 0.7}, # Speed reduction factor
            {'visualization_delay': 0.2},
            {'speed_scaling_factor': 0.8},
            {'min_sync_distance': 0.3},
            {'sync_update_rate': 10.0}
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
        mcts_planner_node,
    ])