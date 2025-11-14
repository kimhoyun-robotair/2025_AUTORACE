import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    pkg_dir = get_package_share_directory('razor_imu_9dof_ros2')
    config_file = os.path.join(pkg_dir, 'config', 'my_razor.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'razor_config_file',
            default_value=config_file,
            description='Path to the razor config file'
        ),
        
        Node(
            package='razor_imu_9dof_ros2',
            executable='imu_node',
            name='razor_node',
            output='screen',
            parameters=[LaunchConfiguration('razor_config_file')]
        )
    ])
