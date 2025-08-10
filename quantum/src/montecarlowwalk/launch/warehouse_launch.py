import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, LifecycleNode
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Paths
    gazebo_ros_pkg = get_package_share_directory("gazebo_ros")
    warehouse_sim_pkg = get_package_share_directory("montecarlowwalk")
    turtlebot3_pkg = get_package_share_directory("turtlebot3_gazebo")

    # World file
    world_file = PathJoinSubstitution([warehouse_sim_pkg, "worlds", "warehouse.world"])

    # Robot model file (if applicable)
    model_file = PathJoinSubstitution([turtlebot3_pkg, "models", "turtlebot3_burger", "model.sdf"])

    # Map file
    map_file = "/home/kabita/quantum/src/montecarlowwalk/maps/map.yaml"
    globalcost = "/home/kabita/ros2_ws/src/warehouse_sim/config/global_costmap.yaml"
    localcost = "/home/kabita/ros2_ws/src/warehouse_sim/config/local_costmap.yaml"
    
    print("Map file path:", map_file)  # Debug print

    # Gazebo launch
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_pkg, "launch", "gazebo.launch.py")
        ),
        launch_arguments={"world": world_file}.items(),
    )

    # Spawn the robot
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
    rviz_config_file = PathJoinSubstitution([warehouse_sim_pkg, "rviz", "warehouse.rviz"])
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
        parameters=[{"use_sim_time": True}],
        arguments=[PathJoinSubstitution([turtlebot3_pkg, "urdf", "turtlebot3_burger.urdf"])]
    )

    # Joint state publisher
    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        output="screen",
        parameters=[{"use_sim_time": True}]
    )

    # Map server (for loading a pre-built map)
    map_server = LifecycleNode(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        namespace="",  # Set namespace to an empty string
        output="screen",
        parameters=[{"yaml_filename": map_file, "use_sim_time": True}],
    )

    # Lifecycle manager for map_server
    lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager",
        output="screen",
        parameters=[{"autostart": True,"use_sim_time": True}],
        arguments=["--ros-args", "--log-level", "info"],
    )

    # Global and Local Costmap Nodes
    global_costmap =Node(
            package='nav2_costmap_2d',
            executable='nav2_costmap_2d',
            name='global_costmap',
            output='screen',
            parameters=[{'use_sim_time': True, 'params_file': '/home/kabita/ros2_ws/src/warehouse_sim/config/global_costmap.yaml'}]
    )

    local_costmap = Node(
            package='nav2_costmap_2d',
            executable='nav2_costmap_2d',
            name='local_costmap',
            output='screen',
            parameters=[{'use_sim_time': True, 'params_file': '/home/kabita/ros2_ws/src/warehouse_sim/config/local_costmap.yaml'}]
        )
    bt_nav = Node(
            package='nav2_bt_navigator',
            executable='bt_navigator',
            name='bt_navigator',
            output='screen',
            parameters=[{'use_sim_time': True, 'action_server_timeout': 10000}]
        )
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument("world", default_value=world_file, description="Path to the warehouse world file"),

        # Launch Gazebo
        gazebo_launch,

        # Spawn the robot
        robot_spawn,

        # Launch RViz
        rviz_node,

        # Robot state publisher
        robot_state_publisher,

        # Joint state publisher
        joint_state_publisher,

        # Map server
        map_server,

        # Lifecycle manager
        lifecycle_manager,
        bt_nav,
        # Global and Local Costmap Nodes
        global_costmap,
        local_costmap,
    ])