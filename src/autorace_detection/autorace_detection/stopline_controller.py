#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32


class StoplineController(Node):
    def __init__(self) -> None:
        super().__init__("stopline_controller")
        self.declare_parameter("stop_min_pixels", 800)
        self.declare_parameter("reverse_scale", 2.0)

        self.bridge = CvBridge()
        self.current_speed = 0.0

        self.create_subscription(Image, "/lane/white_mask", self.mask_cb, 10)
        self.create_subscription(Float32, "/fsm/current_speed", self.speed_cb, 10)
        self.detect_pub = self.create_publisher(Bool, "/stopline/detected", 10)
        self.cmd_pub = self.create_publisher(Twist, "/stopline/reverse_cmd", 10)

        self.get_logger().info("StoplineController started; listening to /lane/white_mask")

    def speed_cb(self, msg: Float32) -> None:
        self.current_speed = float(msg.data)

    def mask_cb(self, msg: Image) -> None:
        try:
            mask_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as err:  # noqa: BLE001
            self.get_logger().warn(f"StoplineController cv_bridge failed: {err}")
            return

        gray = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)
        pixels = int(cv2.countNonZero(gray))
        detected = pixels >= int(self.get_parameter("stop_min_pixels").value)
        self.detect_pub.publish(Bool(data=detected))

        if detected:
            twist = Twist()
            twist.linear.x = -abs(self.current_speed) * float(
                self.get_parameter("reverse_scale").value
            )
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = StoplineController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
