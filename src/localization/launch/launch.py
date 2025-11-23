# launch/launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution

def generate_launch_description():
    pkg_share = FindPackageShare('localization')

    tf_yaml = PathJoinSubstitution([pkg_share, 'config', 'tf_config.yaml'])
    cam_yaml = PathJoinSubstitution([pkg_share, 'config', 'camera_config.yaml'])
    usb_cam_yaml = PathJoinSubstitution([pkg_share, 'config', 'usb_cam_params.yaml'])

    return LaunchDescription([
        Node(
            package='localization',
            executable='static_tf_node',
            name='static_tf_node',
            output='screen',
            parameters=[{'tf_config_path': tf_yaml}],
        ),
        Node(
            package='localization',
            executable='bev_node',
            name='bev_node',
            output='screen',
            parameters=[{
                'config_path': cam_yaml,
                'base_frame': 'base_link',
                'camera_frame': 'camera_optical_frame',
                #'camera_frame': 'left_camera_link_optical',
                'bev_frame': 'bev',
            }],
        ),
        Node(
            package='usb_cam',
            executable='usb_cam_node_exe',
            name='usb_cam',
            # 위에서 정의한 cam_yaml 변수를 그대로 사용
            parameters=[usb_cam_yaml], 
            output='screen'
        )
    ])

