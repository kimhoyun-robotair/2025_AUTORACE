#!/usr/bin/env python3
"""
Stop-line guard node.
- Subscribes: mask_topic (default: /lane/white_mask from stop_line_detector.py)
- Publishes:  /detections/stopline (Bool) and /cmd_vel (Twist=0 when stopline detected)
Logic:
  - Count white pixels in a bottom ROI (default: bottom 1/3 of the image).
  - If over threshold, assert stopline_detected and hold zero cmd_vel for a duration.
  - Publishes detection flag at 20 Hz even when False (for easy debugging).
  - Warns if no mask frames are received recently.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2


class StopLineStop(Node):
    def __init__(self):
        super().__init__("stop_line_stop")

        # Parameters
        self.declare_parameter("mask_topic", "/lane/white_mask")
        self.declare_parameter("roi_start_ratio", 0.66)     # ROI start height ratio from top (0-1)
        self.declare_parameter("min_pixels", 300)          # minimum white pixels to trigger
        self.declare_parameter("hold_seconds", 2.0)        # hold stop for this duration

        self.mask_topic = self.get_parameter("mask_topic").value
        self.roi_start_ratio = float(self.get_parameter("roi_start_ratio").value)
        self.min_pixels = int(self.get_parameter("min_pixels").value)
        self.hold_seconds = float(self.get_parameter("hold_seconds").value)

        self.bridge = CvBridge()
        self.stop_until = None
        self.detected = False
        self.last_msg_time = None

        self.create_subscription(Image, self.mask_topic, self.mask_cb, 10)
        self.stop_pub = self.create_publisher(Bool, "/detections/stopline", 10)
        # Note: Do not publish cmd_vel directly here to avoid bus fights; FSM will handle stopping.

        self.timer = self.create_timer(0.05, self.tick)  # 20 Hz enforcement
        self.get_logger().info(f"StopLineStop ready: {self.mask_topic} -> /cmd_vel zero on detection")

    def mask_cb(self, msg: Image):
        self.last_msg_time = self.get_clock().now()
        mask_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        gray = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)
        h, _ = gray.shape
        y0 = int(h * self.roi_start_ratio)
        roi = gray[y0:, :]
        count = int(cv2.countNonZero(roi))

        # Debug overlay for ROI and centroid
        debug = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.line(debug, (0, y0), (debug.shape[1], y0), (0, 255, 255), 2)
        if count > 0:
            m = cv2.moments(roi)
            if m["m00"] > 1e-3:
                cx = int(m["m10"] / m["m00"])
                cy = int(m["m01"] / m["m00"] + y0)  # translate back to full image coords
                cv2.circle(debug, (cx, cy), 6, (0, 255, 0), 2)
        cv2.putText(debug, f"count={count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
        cv2.imshow("stopline_debug", debug)
        cv2.waitKey(1)

        if count >= self.min_pixels:
            self.detected = True
            now = self.get_clock().now()
            self.stop_until = now.nanoseconds + int(self.hold_seconds * 1e9)
            self.get_logger().info(f"Stopline detected: pixels={count}, holding stop for {self.hold_seconds:.1f}s")
        else:
            # do not reset detected flag immediately; rely on hold duration
            if not self.is_holding():
                self.detected = False

    def is_holding(self):
        if self.stop_until is None:
            return False
        return self.get_clock().now().nanoseconds < self.stop_until

    def tick(self):
        # Warn if no mask frames are arriving
        if self.last_msg_time:
            age = (self.get_clock().now() - self.last_msg_time).nanoseconds * 1e-9
            if age > 1.0:
                self.get_logger().warn(f"No mask frames for {age:.1f}s on {self.mask_topic}")
                self.last_msg_time = None  # avoid repeated warnings

        holding = self.is_holding()
        stop_active = self.detected or holding
        # publish detection flag
        self.stop_pub.publish(Bool(data=stop_active))
        # FSM will subscribe to /detections/stopline and handle stopping on /cmd_vel.


def main(args=None):
    rclpy.init(args=args)
    node = StopLineStop()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
