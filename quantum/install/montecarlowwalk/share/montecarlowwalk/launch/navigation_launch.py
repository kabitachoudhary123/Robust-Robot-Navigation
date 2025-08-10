import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true', description='Use simulation time'),
        
        # Launch navigation2
        Node(
            package='nav2_bringup',
            executable='bringup_launch.py',
            name='nav2_bringup',
            output='screen',
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
            remappings=[('/tf', '/tf'), ('/tf_static', '/tf_static')]
        ),
        
        # Add any other necessary nodes or parameters here
    ])
