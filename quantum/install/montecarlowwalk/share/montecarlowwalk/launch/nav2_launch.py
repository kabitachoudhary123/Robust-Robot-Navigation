import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Paths
    montecarlowwalk_pkg = get_package_share_directory("montecarlowwalk")
    nav2_bringup_pkg = get_package_share_directory('nav2_bringup')

    # Nav2 Bringup Launch
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_pkg, 'launch', 'bringup_launch.py')
        ),
        launch_arguments={
            'map': "/home/kabita/ros2_ws/src/warehouse_sim/maps/map.yaml",
            'use_sim_time': 'True',
            #'params_file': os.path.join(montecarlowwalk_pkg, 'config', 'nav2_params.yaml')
        }.items()
    )

    return LaunchDescription([
        nav2_launch,
    ])
