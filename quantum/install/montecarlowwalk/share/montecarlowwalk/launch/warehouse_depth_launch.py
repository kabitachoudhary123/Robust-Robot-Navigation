import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, LifecycleNode
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Package paths
    gazebo_ros_pkg = get_package_share_directory("gazebo_ros")
    warehouse_sim_pkg = get_package_share_directory("warehouse_sim")
    turtlebot3_pkg = get_package_share_directory("turtlebot3_gazebo")

    # File paths
    world_file = PathJoinSubstitution([warehouse_sim_pkg, "worlds", "new_warehouse.world"])
    urdf_file = PathJoinSubstitution([turtlebot3_pkg, "urdf", "turtlebot3_burger.urdf"])
    rviz_config_file = PathJoinSubstitution([warehouse_sim_pkg, "rviz", "warehouse.rviz"])
    model_file = "/home/kabita/turtlebot3_custom_models/turtlebot3_burger/model.sdf"
    map_file = "/home/kabita/ros2_ws/newwarehousemap/map.yaml"

    # Gazebo launch
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_pkg, "launch", "gazebo.launch.py")
        ),
        launch_arguments={"world": world_file, "verbose": "true"}.items(),
    )

    # Spawn the robot (with small Z offset to prevent collision with ground)
    robot_spawn = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=["-entity", "turtlebot3_burger", "-file", model_file, "-x", "0.0", "-y", "0.0", "-z", "0.1"],
        output="screen"
    )

    # Robot state publisher
    robot_state_publisher = Node(
    package="robot_state_publisher",
    executable="robot_state_publisher",
    parameters=[{
        "use_sim_time": True,
        "robot_description": open(model_file, 'r').read()  # If SDF can work here
    }],
    output="screen"
)

    # Joint state publisher
    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        parameters=[{"use_sim_time": True}],
        output="screen"
    )

    # RViz2
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", rviz_config_file],
        parameters=[{"use_sim_time": True}],
        output="screen"
    )

    # Map server (loads static map)
    map_server = LifecycleNode(
    package="nav2_map_server",
    executable="map_server",
    name="map_server",
    namespace="",  # <-- REQUIRED even if empty
    output="screen",
    parameters=[{"yaml_filename": map_file, "use_sim_time": True}],
)


    # AMCL localization node
    amcl_node = Node(
        package="nav2_amcl",
        executable="amcl",
        name="amcl",
        namespace="", 
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "min_particles": 500,
            "max_particles": 2000,
            "resample_interval": 1,
            "transform_tolerance": 0.2,
            "laser_min_range": 0.1,
            "laser_max_range": 3.5,
        }],
    )

    # Lifecycle manager
    lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_map",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "autostart": True,
            "node_names": ["map_server", "amcl"]
        }]
    )

    return LaunchDescription([
        # Declare world path as argument (optional)
        DeclareLaunchArgument("world", default_value=world_file, description="Gazebo world file"),

        # Launch Gazebo
        gazebo_launch,

        # Delay robot spawn slightly (if needed)
        TimerAction(period=5.0, actions=[robot_spawn]),

        # Robot state and joint publisher
        robot_state_publisher,
        joint_state_publisher,

        # Static map server and AMCL
        map_server,
        amcl_node,
        lifecycle_manager,

        # RViz
        TimerAction(period=6.0, actions=[rviz_node]),
    ])
