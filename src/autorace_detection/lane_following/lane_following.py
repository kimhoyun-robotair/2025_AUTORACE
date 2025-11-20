#!/usr/bin/env python3
"""
Lane following controller.
- Subscribes: /lane/white_mask (from lane_detector.py, BEV mask BGR image)
- Publishes:  /control/color_zone_twist (Twist command proposal for FSM COLOR_ZONE)
Strategy:
  - Compute mask centroid (cx) and steer based on offset from image center.
  - If mask is lost for a few frames, stop for safety.
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge


class LaneFollower(Node):
    def __init__(self):
        super().__init__("lane_follower")

        # Parameters for simple steering control
        self.declare_parameter("linear_speed", 0.35)
        self.declare_parameter("angular_gain", 2.0)       # gain on normalized error
        self.declare_parameter("max_angular", 0.7)        # max angular speed
        self.declare_parameter("center_bias_px", 0.0)     # pixel bias if camera is offset
        self.declare_parameter("lost_stop_frames", 5)     # consecutive lost frames before stop

        self.linear_speed = float(self.get_parameter("linear_speed").value)
        self.angular_gain = float(self.get_parameter("angular_gain").value)
        self.max_angular = float(self.get_parameter("max_angular").value)
        self.center_bias_px = float(self.get_parameter("center_bias_px").value)
        self.lost_stop_frames = int(self.get_parameter("lost_stop_frames").value)

        self.bridge = CvBridge()
        self.lost_counter = 0

        self.create_subscription(Image, "/lane/yellow_mask", self.mask_cb, 10)
        # Publish to a dedicated lane-following control channel; FSM will republish to /cmd_vel.
        # (Changed from publishing directly to /cmd_vel to avoid bus contention.)
        self.cmd_pub = self.create_publisher(Twist, "/control/lane_cmd", 10)

        self.get_logger().info("LaneFollower ready: sub /lane/white_mask -> pub /control/lane_cmd")

    def mask_cb(self, msg: Image):
        # Convert to grayscale mask
        mask_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        gray = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)

        m = cv2.moments(gray)
        if m["m00"] < 1e-3:
            self.lost_counter += 1
            if self.lost_counter >= self.lost_stop_frames:
                self._publish_stop()
            return

        self.lost_counter = 0
        cx = m["m10"] / m["m00"]
        width = gray.shape[1]
        target = width / 2.0 + self.center_bias_px
        error = (target - cx) / width  # normalized offset

        twist = Twist()
        twist.linear.x = self.linear_speed
        twist.angular.z = float(np.clip(self.angular_gain * error, -self.max_angular, self.max_angular))
        self.cmd_pub.publish(twist)

    def _publish_stop(self):
        twist = Twist()
        self.cmd_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = LaneFollower()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
