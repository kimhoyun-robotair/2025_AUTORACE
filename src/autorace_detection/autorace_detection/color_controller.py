#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String


class ColorBasedDrive(Node):
    def __init__(self):
        super().__init__("color_based_drive")

        self.declare_parameter("speed_red", 0.2)
        self.declare_parameter("speed_blue", 0.5)

        self.speed_red = float(self.get_parameter("speed_red").value)
        self.speed_blue = float(self.get_parameter("speed_blue").value)

        self.current_color = "unknown"

        # Publish to a dedicated color-control channel; FSM will republish to /cmd_vel.
        self.cmd_pub = self.create_publisher(Twist, "/color/cmd_vel", 10)
        self.create_subscription(String, "/lane/color", self.on_color, 10)

        self.get_logger().info("ColorBasedDrive ready: /lane/color -> /color/cmd_vel (straight only)")

    def on_color(self, msg: String):
        self.current_color = msg.data.lower().strip()
        self.publish_cmd()

    def publish_cmd(self):
        twist = Twist()
        if self.current_color == "red":
            twist.linear.x = self.speed_red
        elif self.current_color == "blue":
            twist.linear.x = self.speed_blue
        else:
            twist.linear.x = 0.2
        twist.angular.z = 0.0
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
