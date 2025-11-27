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
    sim_share = get_package_share_directory("autorace_sim_classic")
    spawn_launch = os.path.join(sim_share, "launch", "spawn_robot.launch.py")

    image_share = get_package_share_directory('image_preprocessing')
    bev_launch = os.path.join(image_share, 'launch', 'image_preprocessing.launch.py')

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
    ]

    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(spawn_launch),
                launch_arguments={}.items(),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(bev_launch),
                launch_arguments={}.items(),
            ),
        ]
        + detection_nodes
    )
