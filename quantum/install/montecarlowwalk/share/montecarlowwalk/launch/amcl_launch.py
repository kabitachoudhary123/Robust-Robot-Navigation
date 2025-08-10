from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # AMCL Node
        Node(
            package='nav2_amcl',
            executable='amcl',
            name='amcl',
            output='screen',
            parameters=[{'use_sim_time': True}]
        ),

        # Lifecycle Manager Node
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager',
            output='screen',
            parameters=[{
                'autostart': True,
                'node_names': ['amcl']
            }]
        ),
    ])