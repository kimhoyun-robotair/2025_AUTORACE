#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image


class YellowLaneDetector(Node):
    """
    Subscribes to BEV images, isolates yellow lanes with a single robust step,
    and publishes a binary mask plus an overlay for visualization.
    """

    def __init__(self) -> None:
        super().__init__("yellow_lane_detector")
        self.bridge = CvBridge()
        self.create_subscription(
            Image,
            "/image_bev",
            self.image_callback,
            10,
        )
        self.declare_parameter("crop_top", 0.2)     # fraction of height (0.0~1.0)
        self.declare_parameter("crop_bottom", 0.0)  # fraction of height (0.0~1.0)
        self.declare_parameter("crop_left", 0.17)    # fraction of width  (0.0~1.0)
        self.declare_parameter("crop_right", 0.21)   # fraction of width  (0.0~1.0)
        self.mask_pub = self.create_publisher(Image, "/yellow_lane_mask", 10)
        self.overlay_pub = self.create_publisher(Image, "/yellow_lane_overlay", 10)
        self.get_logger().info("Yellow lane detector node started.")

    def image_callback(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as err:
            self.get_logger().warn(f"cv_bridge conversion failed: {err}")
            return

        frame = self._crop_frame(frame)

        # Single preprocessing step: move to CIE LAB space and threshold the b-channel
        # to isolate yellow, which remains high on the blue-yellow axis even under neon.
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        b_channel = lab[:, :, 2]
        adaptive_threshold = np.clip(np.percentile(b_channel, 88), 150, 245)
        _, mask = cv2.threshold(b_channel, adaptive_threshold, 255, cv2.THRESH_BINARY)

        overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(frame, 0.8, overlay, 0.6, 0.0)

        moments = cv2.moments(mask)
        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            cv2.circle(overlay, (cx, cy), 6, (0, 255, 0), -1)
            cv2.line(
                overlay, (cx, cy), (cx, overlay.shape[0] - 1), (0, 255, 0), 2
            )

        self.mask_pub.publish(self.bridge.cv2_to_imgmsg(mask, encoding="mono8"))
        self.overlay_pub.publish(self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8"))

        # Optional local visualization for quick verification (binary mask).
        try:
            cv2.imshow("yellow_lane_mask", mask)
            cv2.waitKey(1)
        except cv2.error as err:
            self.get_logger().warn(f"cv2.imshow failed (GUI may be unavailable): {err}")

    def _crop_frame(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        top_f = max(0.0, self.get_parameter("crop_top").get_parameter_value().double_value)
        bottom_f = max(0.0, self.get_parameter("crop_bottom").get_parameter_value().double_value)
        left_f = max(0.0, self.get_parameter("crop_left").get_parameter_value().double_value)
        right_f = max(0.0, self.get_parameter("crop_right").get_parameter_value().double_value)

        # Clamp to [0, 0.95] to avoid degenerate cases while allowing generous cropping.
        top_f = min(top_f, 0.95)
        bottom_f = min(bottom_f, 0.95)
        left_f = min(left_f, 0.95)
        right_f = min(right_f, 0.95)

        if top_f + bottom_f >= 0.98 or left_f + right_f >= 0.98:
            self.get_logger().warn(
                "Crop fractions too large (sum >= 0.98); skipping crop this frame."
            )
            return frame

        top_px = int(round(top_f * h))
        bottom_px = int(round(bottom_f * h))
        left_px = int(round(left_f * w))
        right_px = int(round(right_f * w))

        return frame[top_px : h - bottom_px, left_px : w - right_px]


def main(args=None) -> None:
    rclpy.init(args=args)
    node = YellowLaneDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        rclpy.shutdown()


if __name__ == "__main__":
    main()
