#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import math
import numpy as np

class Lidar180Node(Node):
    def __init__(self):
        super().__init__('lidar_180_node')

        # 원본 scan 구독
        self.sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # 보정된 scan 퍼블리셔
        self.pub = self.create_publisher(
            LaserScan,
            '/scan_rotated',
            10
        )

        # TF broadcaster
        self.br = TransformBroadcaster(self)

        self.get_logger().info("Lidar 180-degree correction node started")

    def scan_callback(self, msg):
        # 새로운 LaserScan 메세지 생성
        corrected = LaserScan()
        corrected.header = msg.header
        corrected.header.frame_id = 'lidar_rotated'

        # *** 핵심: 180도 회전 처리 ***
        corrected.angle_min = msg.angle_min + math.pi
        corrected.angle_max = msg.angle_max + math.pi
        corrected.angle_increment = msg.angle_increment
        corrected.time_increment = msg.time_increment
        corrected.scan_time = msg.scan_time
        corrected.range_min = msg.range_min
        corrected.range_max = msg.range_max

        # ranges 배열 반전 (물리 180° 회전)
        corrected.ranges = list(msg.ranges[::-1])

        # intensities 도 반전
        corrected.intensities = list(msg.intensities[::-1])

        self.pub.publish(corrected)

        # TF도 함께 브로드캐스트
        self.publish_tf(msg.header.stamp)

    def publish_tf(self, stamp):
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = 'laser'             # 원본 라이다 프레임
        t.child_frame_id = 'lidar_rotated'    # 회전된 프레임

        # 180도 회전 (Z축 기준)
        q = self.quaternion_from_yaw(math.pi)

        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.br.sendTransform(t)

    def quaternion_from_yaw(self, yaw):
        return [
            0.0,
            0.0,
            math.sin(yaw / 2.0),
            math.cos(yaw / 2.0)
        ]

def main(args=None):
    rclpy.init(args=args)
    node = Lidar180Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

