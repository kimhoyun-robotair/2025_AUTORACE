#!/usr/bin/env python3
"""
Color-based speed + heading controller.
- Subscribes: /lane/color (String: "red" | "blue" | other)
- Subscribes: /lane/color_center_offset (Float32: (cx - w/2)/w, +->right)
- Publishes:  /cmd_vel (Twist)
Policy:
  red  -> linear.x = speed_red
  blue -> linear.x = speed_blue
  else -> stop (0)
  angular.z = gain * offset, clamped to max_angular
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float32
import numpy as np


class ColorBasedDrive(Node):
    def __init__(self):
        super().__init__("color_based_drive")

        self.declare_parameter("speed_red", 0.2)
        self.declare_parameter("speed_blue", 0.5)
        self.declare_parameter("angular_gain", 1.0)      # lower gain to reduce oscillation
        self.declare_parameter("max_angular", 0.2)       # lower max angular speed
        self.declare_parameter("deadband", 0.02)         # normalized offset deadband

        self.speed_red = float(self.get_parameter("speed_red").value)
        self.speed_blue = float(self.get_parameter("speed_blue").value)
        self.angular_gain = float(self.get_parameter("angular_gain").value)
        self.max_angular = float(self.get_parameter("max_angular").value)
        self.deadband = float(self.get_parameter("deadband").value)

        self.current_color = "unknown"
        self.center_offset = 0.0

        # Publish to a dedicated color-control channel; FSM will republish to /cmd_vel.
        # (Changed from publishing directly to /cmd_vel to avoid bus contention.)
        self.cmd_pub = self.create_publisher(Twist, "/control/color_cmd", 10)
        self.create_subscription(String, "/lane/color", self.on_color, 10)
        self.create_subscription(Float32, "/lane/color_center_offset", self.on_center, 10)

        self.get_logger().info("ColorBasedDrive ready: /lane/color + center -> /control/color_cmd")

    def on_color(self, msg: String):
        self.current_color = msg.data.lower().strip()
        self.publish_cmd()

    def on_center(self, msg: Float32):
        self.center_offset = msg.data
        self.publish_cmd()

    def publish_cmd(self):
        twist = Twist()
        if self.current_color == "red":
            twist.linear.x = self.speed_red
        elif self.current_color == "blue":
            twist.linear.x = self.speed_blue
        else:
            twist.linear.x = 0.0

        offset = 0.0 if abs(self.center_offset) < self.deadband else self.center_offset
        twist.angular.z = float(np.clip(self.angular_gain * offset,
                                        -self.max_angular, self.max_angular))
        self.cmd_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = ColorBasedDrive()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
