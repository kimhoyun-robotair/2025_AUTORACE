#!/usr/bin/env python3
"""
Bringup launch:
- Include autorace_sim_classic spawn_robot.launch.py
- Start autorace_detection nodes used so far:
  lane_detector.py, lane_following.py, color_detection.py,
  color_based_driving.py, stop_line_detector.py, stop_line_stop.py
- mission_fsm.py intentionally NOT launched here.
"""

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    image_share = get_package_share_directory('image_preprocessing')
    bev_launch = os.path.join(image_share, 'launch', 'image_preprocessing.launch.py')

    pkg_f1tenth = get_package_share_directory('f1tenth_stack')
    launch_vesc = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_f1tenth, 'launch', 'bringup_launch.py')
        )
    )

    pkg_lidar = get_package_share_directory('rplidar_ros') 
    launch_lidar = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_lidar, 'launch', 'rplidar_s1_launch.py') 
        )
    )

    pkg_imu = get_package_share_directory('razor_imu_9dof_ros2') 
    launch_imu = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_imu, 'launch', 'razor.launch.py') 
        )
    )

    detection_nodes = [
        Node(
            package="autorace_detection",
            executable="lane_detection",
            name="lane_detector",
            output="screen",
        ),
        Node(
            package="autorace_detection",
            executable="lane_controller",
            name="lane_following",
            output="screen",
        ),
        Node(
            package="autorace_detection",
            executable="red_blue_detector",
            name="color_detection",
            output="screen",
        ),
        Node(
            package="autorace_detection",
            executable="color_controller",
            name="color_based_driving",
            output="screen",
        ),
        Node(
            package="autorace_detection",
            executable="stopline_detection",
            name="stop_line_detector",
            output="screen",
        ),
        Node(
            package="autorace_detection",
            executable="stopline_controller",
            name="stop_line_stop",
            output="screen",
        ),
        Node(
            package="autorace_bringup",
            executable="twist_to_ackermann",
            name="twist_to_ackermann",
            output="screen",
        ),
    ]

    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(bev_launch),
                launch_arguments={}.items(),
            ),
            launch_imu,
            launch_lidar,
            launch_vesc,
        ]
        # + detection_nodes
    )
