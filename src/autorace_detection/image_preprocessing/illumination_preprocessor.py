#!/usr/bin/env python3
"""
Glare/over-exposure reducer before lane/color detection.
- Subscribes to an input image topic (default: /image_raw)
- Applies brightness suppression, optional CLAHE contrast recovery, optional gamma correction
- Republishes the corrected image for downstream detectors
"""

import rclpy
from rclpy.node import Node

import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class IlluminationPreprocessor(Node):
    def __init__(self):
        super().__init__('illumination_preprocessor')

        self.declare_parameter('input_image_topic', '/image_bev')
        self.declare_parameter('output_image_topic', '/image_balanced')
        self.declare_parameter('value_scale', 0.45)          # <1.0 darkens highlights
        self.declare_parameter('saturation_scale', 1.0)      # Optional color boost
        self.declare_parameter('use_clahe', True)            # Contrast Limited AHE on V channel
        self.declare_parameter('clahe_clip_limit', 2.0)
        self.declare_parameter('clahe_tile_grid_size', 12)
        self.declare_parameter('use_gamma', True)           # Gamma correction on V channel
        self.declare_parameter('gamma', 2.0)                 # >1 darkens, <1 brightens here
        self.declare_parameter('blur_kernel_size', 5)        # Mild blur to soften specular noise
        self.declare_parameter('show_debug', True)

        self.update_params_from_server()

        self.bridge = CvBridge()
        self.sub = self.create_subscription(
            Image,
            self.input_image_topic,
            self.image_callback,
            10
        )
        self.pub = self.create_publisher(
            Image,
            self.output_image_topic,
            10
        )
        self.get_logger().info(
            f'IlluminationPreprocessor subscribed to {self.input_image_topic}, '
            f'publishing to {self.output_image_topic}'
        )

    def update_params_from_server(self):
        self.input_image_topic = self.get_parameter('input_image_topic').value
        self.output_image_topic = self.get_parameter('output_image_topic').value
        self.value_scale = float(self.get_parameter('value_scale').value)
        self.saturation_scale = float(self.get_parameter('saturation_scale').value)
        self.use_clahe = bool(self.get_parameter('use_clahe').value)
        self.clahe_clip_limit = float(self.get_parameter('clahe_clip_limit').value)
        self.clahe_tile_grid_size = max(1, int(self.get_parameter('clahe_tile_grid_size').value))
        self.use_gamma = bool(self.get_parameter('use_gamma').value)
        self.gamma = max(0.01, float(self.get_parameter('gamma').value))
        self.blur_kernel_size = int(self.get_parameter('blur_kernel_size').value)
        self.show_debug = bool(self.get_parameter('show_debug').value)
        self.gamma_lut = self._build_gamma_lut(self.gamma)

        if self.blur_kernel_size % 2 == 0:
            self.blur_kernel_size = max(1, self.blur_kernel_size - 1)

    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        processed = self.preprocess_image(frame)

        out_msg = self.bridge.cv2_to_imgmsg(processed, encoding='bgr8')
        out_msg.header = msg.header
        self.pub.publish(out_msg)

        if self.show_debug:
            cv2.imshow('illumination_preprocessor', processed)
            cv2.waitKey(1)

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Suppress glare, recover contrast, and stabilize color saturation."""
        work = image

        if self.blur_kernel_size > 1:
            work = cv2.GaussianBlur(work, (self.blur_kernel_size, self.blur_kernel_size), 0)

        hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        if self.value_scale != 1.0:
            v = np.clip(v.astype(np.float32) * self.value_scale, 0, 255).astype(np.uint8)

        if self.use_clahe:
            clahe = cv2.createCLAHE(
                clipLimit=max(0.1, self.clahe_clip_limit),
                tileGridSize=(self.clahe_tile_grid_size, self.clahe_tile_grid_size)
            )
            v = clahe.apply(v)

        if self.use_gamma and abs(self.gamma - 1.0) > 1e-3:
            v = cv2.LUT(v, self.gamma_lut)

        if self.saturation_scale != 1.0:
            s = np.clip(s.astype(np.float32) * self.saturation_scale, 0, 255).astype(np.uint8)

        hsv = cv2.merge((h, s, v))
        balanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return balanced

    @staticmethod
    def _build_gamma_lut(gamma: float) -> np.ndarray:
        inv_gamma = 1.0 / gamma
        lut = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)], dtype=np.float32)
        return np.clip(lut, 0, 255).astype("uint8")


def main(args=None):
    rclpy.init(args=args)
    node = IlluminationPreprocessor()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
