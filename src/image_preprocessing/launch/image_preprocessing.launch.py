# launch/launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution

def generate_launch_description():
    pkg_share = FindPackageShare('image_preprocessing')

    tf_yaml = PathJoinSubstitution([pkg_share, 'config', 'tf_config.yaml'])
    cam_yaml = PathJoinSubstitution([pkg_share, 'config', 'camera_config.yaml'])

    return LaunchDescription([
        Node(
            package='image_preprocessing',
            executable='static_tf_node',
            name='static_tf_node',
            output='screen',
            parameters=[{'tf_config_path': tf_yaml}],
        ),
        Node(
            package='image_preprocessing',
            executable='bev_node',
            name='bev_node',
            output='screen',
            parameters=[{
                'config_path': cam_yaml,
                'base_frame': 'base_link',
                # 'camera_frame': 'camera_optical_frame',
                'camera_frame': 'left_camera_link_optical', # For Simulation
                'bev_frame': 'bev',
            }],
        ),
    ])

