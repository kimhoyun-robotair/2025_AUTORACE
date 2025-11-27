#!/usr/bin/env python3
"""
Lane following controller.
- Subscribes: /yellow_lane_mask (from lane_detector.py, BEV mask BGR image)
- Publishes:  /lane/cmd_vel (Twist proposal for FSM)
- Publishes:  /lane/has_lane (Bool) for validity gating
Strategy:
  - Compute mask centroid (cx) and steer based on offset from image center.
  - If mask is lost for a few frames, stop for safety.
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge


class LaneFollower(Node):
    def __init__(self):
        super().__init__("lane_follower")

        # Parameters for simple steering control
        self.declare_parameter("linear_speed", 0.3)
        self.declare_parameter("angular_gain", 2.5)       # gain on normalized error
        self.declare_parameter("max_angular", 0.3)        # max angular speed
        self.declare_parameter("center_bias_px", 20.0)     # pixel bias if camera is offset
        self.declare_parameter("lost_stop_frames", 5)     # consecutive lost frames before stop

        self.linear_speed = float(self.get_parameter("linear_speed").value)
        self.angular_gain = float(self.get_parameter("angular_gain").value)
        self.max_angular = float(self.get_parameter("max_angular").value)
        self.center_bias_px = float(self.get_parameter("center_bias_px").value)
        self.lost_stop_frames = int(self.get_parameter("lost_stop_frames").value)

        self.bridge = CvBridge()
        self.lost_counter = 0

        self.create_subscription(Image, "/yellow_lane_mask", self.mask_cb, 10)
        # Publish to a dedicated lane-following control channel; FSM will republish to /cmd_vel.
        # (Changed from publishing directly to /cmd_vel to avoid bus contention.)
        self.cmd_pub = self.create_publisher(Twist, "/lane/cmd_vel", 10)
        self.valid_pub = self.create_publisher(Bool, "/lane/has_lane", 10)
        self.get_logger().info("LaneFollower ready: sub /yellow_lane_mask -> pub /lane/cmd_vel + /lane/has_lane")

    def mask_cb(self, msg: Image):
        # Convert to grayscale mask
        mask_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        gray = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)

        m = cv2.moments(gray)
        if m["m00"] < 1e-3:
            self.lost_counter += 1
            self.valid_pub.publish(Bool(data=False))
            if self.lost_counter >= self.lost_stop_frames:
                self._publish_stop()
            return

        self.lost_counter = 0
        self.valid_pub.publish(Bool(data=True))
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
