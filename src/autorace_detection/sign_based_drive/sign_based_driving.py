#!/usr/bin/env python3
"""
Sign-based driving node.
- Subscribes: YOLO detections (from yolo_ros) to pick left/right sign.
- Subscribes: lane mask image (BEV) to run lane following biased to that side.
- Publishes:  Twist command.
Strategy:
  1) Listen to yolo_msgs/DetectionArray, keep freshest left/right sign above a score threshold.
  2) On every lane mask frame, crop the mask to the chosen side, find centroid, and steer toward it.
  3) If detections are stale or the mask is lost, stop for safety.
"""

from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import Image
from yolo_msgs.msg import DetectionArray


class SignBasedDriving(Node):
    def __init__(self) -> None:
        super().__init__("sign_based_driving")

        # Topics
        self.declare_parameter("detection_topic", "/detections")
        self.declare_parameter("lane_mask_topic", "/lane/yellow_mask")
        self.declare_parameter("cmd_topic", "/cmd_vel")

        # Driving gains
        self.declare_parameter("linear_speed", 0.5)
        self.declare_parameter("angular_gain", 2.0)
        self.declare_parameter("max_angular", 0.7)
        self.declare_parameter("center_bias_px", 0.0)
        self.declare_parameter("lost_stop_frames", 5)

        # Sign filtering / ROI selection
        self.declare_parameter("min_detection_score", 0.4)
        self.declare_parameter("direction_timeout", 2.0)  # seconds before sign is considered stale
        self.declare_parameter("roi_ratio", 0.6)          # portion of the image width to use (0-1], per side

        self.detection_topic = self.get_parameter("detection_topic").value
        self.lane_mask_topic = self.get_parameter("lane_mask_topic").value
        self.cmd_topic = self.get_parameter("cmd_topic").value

        self.linear_speed = float(self.get_parameter("linear_speed").value)
        self.angular_gain = float(self.get_parameter("angular_gain").value)
        self.max_angular = float(self.get_parameter("max_angular").value)
        self.center_bias_px = float(self.get_parameter("center_bias_px").value)
        self.lost_stop_frames = int(self.get_parameter("lost_stop_frames").value)

        self.min_detection_score = float(self.get_parameter("min_detection_score").value)
        self.direction_timeout = float(self.get_parameter("direction_timeout").value)
        self.roi_ratio = float(self.get_parameter("roi_ratio").value)
        self.roi_ratio = np.clip(self.roi_ratio, 0.05, 1.0)  # avoid degenerate crop

        self.bridge = CvBridge()
        self.current_direction: Optional[str] = None  # "left" | "right"
        self.last_direction_stamp = None
        self.lost_counter = 0

        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)
        self.create_subscription(DetectionArray, self.detection_topic, self.on_detections, 10)
        self.create_subscription(Image, self.lane_mask_topic, self.on_mask, 10)

        self.get_logger().info(
            f"SignBasedDriving ready: signs from {self.detection_topic}, mask from "
            f"{self.lane_mask_topic}, cmd -> {self.cmd_topic}"
        )

    # ----------------- Callbacks -----------------

    def on_detections(self, msg: DetectionArray) -> None:
        """Pick the strongest left/right detection and store it."""
        best_direction = None
        best_score = self.min_detection_score

        for det in msg.detections:
            name = det.class_name.lower()
            if name not in ("left", "right"):
                continue
            if det.score >= best_score:
                best_direction = name
                best_score = det.score

        if best_direction is None:
            return

        direction_changed = best_direction != self.current_direction
        self.current_direction = best_direction
        self.last_direction_stamp = self.get_clock().now()

        if direction_changed:
            self.get_logger().info(f"Detected turn sign: {self.current_direction} (score {best_score:.2f})")

    def on_mask(self, msg: Image) -> None:
        """Lane following biased to the sign direction."""
        direction = self.get_active_direction()
        if direction is None:
            self.publish_stop()
            return

        mask_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        gray = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        roi, x_offset = self.select_roi(gray, direction)
        m = cv2.moments(roi)
        if m["m00"] < 1e-3:
            self.lost_counter += 1
            if self.lost_counter >= self.lost_stop_frames:
                self.publish_stop()
            return

        self.lost_counter = 0
        cx_local = m["m10"] / m["m00"]
        cx_global = x_offset + cx_local

        target = width / 2.0 + self.center_bias_px
        error = (target - cx_global) / float(width)

        twist = Twist()
        twist.linear.x = self.linear_speed
        twist.angular.z = float(np.clip(self.angular_gain * error, -self.max_angular, self.max_angular))
        self.cmd_pub.publish(twist)

    # ----------------- Helpers -----------------

    def get_active_direction(self) -> Optional[str]:
        """Return current direction if not stale, else None."""
        if self.current_direction is None or self.last_direction_stamp is None:
            return None
        elapsed = self.get_clock().now() - self.last_direction_stamp
        if elapsed > Duration(seconds=self.direction_timeout):
            return None
        return self.current_direction

    def select_roi(self, gray: np.ndarray, direction: str) -> Tuple[np.ndarray, int]:
        """Pick left or right slice of the mask; return ROI and its x-offset."""
        width = gray.shape[1]
        roi_width = max(1, int(width * self.roi_ratio))

        if direction == "left":
            x_start = 0
        else:
            x_start = width - roi_width

        x_end = x_start + roi_width
        return gray[:, x_start:x_end], x_start

    def publish_stop(self) -> None:
        twist = Twist()
        self.cmd_pub.publish(twist)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SignBasedDriving()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
