#!/usr/bin/env python3
"""
Glare/over-exposure reducer before lane/color detection.
- Subscribes to an input image topic (default: /image_bev)
- Applies highlight clipping + color/contrast normalization
- Republishes the corrected image for downstream detectors
Tuning guide:
  - highlight_clip_percentile: raise (e.g., 97->99) if image becomes too dark; lower if LED streaks persist.
  - value_scale: <1 darkens everything; combine with gamma>1 for bright glare; gamma<1 if image is too dark.
  - clahe_clip_limit/tile_grid_size: increase to recover more lane detail; decrease if noise appears.
  - bilateral_*: use small values to smooth color noise without blurring edges; disable (set to 0) if slow.
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
        # Highlight suppression
        self.declare_parameter('highlight_clip_percentile', 98.0)   # percentile of V to clamp; lower removes more glare
        self.declare_parameter('value_scale', 0.65)                 # <1.0 darkens; for heavy glare start 0.5~0.7
        self.declare_parameter('gamma', 0.6)                        # >1 darkens bright parts; <1 brightens dark scenes
        self.declare_parameter('use_gamma', True)

        # Contrast / color
        self.declare_parameter('use_clahe', True)                   # Contrast Limited AHE on V channel
        self.declare_parameter('clahe_clip_limit', 2.0)
        self.declare_parameter('clahe_tile_grid_size', 12)
        self.declare_parameter('saturation_scale', 1.3)             # Slight boost (1.1~1.3) if yellow lanes look washed out

        # Smoothing
        self.declare_parameter('blur_kernel_size', 3)               # Odd >1; increase if random speckles
        self.declare_parameter('bilateral_d', 0)                    # Set >0 (e.g., 5) to smooth color noise without blurring edges
        self.declare_parameter('bilateral_sigma_color', 50.0)
        self.declare_parameter('bilateral_sigma_space', 25.0)

        # Debug
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
        self.highlight_clip_percentile = float(self.get_parameter('highlight_clip_percentile').value)
        self.value_scale = float(self.get_parameter('value_scale').value)
        self.gamma = max(0.01, float(self.get_parameter('gamma').value))
        self.use_gamma = bool(self.get_parameter('use_gamma').value)
        self.use_clahe = bool(self.get_parameter('use_clahe').value)
        self.clahe_clip_limit = float(self.get_parameter('clahe_clip_limit').value)
        self.clahe_tile_grid_size = max(1, int(self.get_parameter('clahe_tile_grid_size').value))
        self.saturation_scale = float(self.get_parameter('saturation_scale').value)
        self.blur_kernel_size = int(self.get_parameter('blur_kernel_size').value)
        self.bilateral_d = int(self.get_parameter('bilateral_d').value)
        self.bilateral_sigma_color = float(self.get_parameter('bilateral_sigma_color').value)
        self.bilateral_sigma_space = float(self.get_parameter('bilateral_sigma_space').value)
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

        # 1) Clip top highlights to reduce LED streak influence
        clip_p = np.clip(self.highlight_clip_percentile, 50.0, 100.0)
        high_val = np.percentile(v, clip_p)
        if high_val > 0:
            v = np.clip(v, 0, high_val).astype(np.float32)
            v = v / max(1.0, high_val) * 255.0
        v = np.clip(v, 0, 255).astype(np.uint8)

        # 2) Value scaling
        if self.value_scale != 1.0:
            v = np.clip(v.astype(np.float32) * self.value_scale, 0, 255).astype(np.uint8)

        # 3) Contrast recovery
        if self.use_clahe:
            clahe = cv2.createCLAHE(
                clipLimit=max(0.1, self.clahe_clip_limit),
                tileGridSize=(self.clahe_tile_grid_size, self.clahe_tile_grid_size)
            )
            v = clahe.apply(v)

        # 4) Gamma for highlight compression
        if self.use_gamma and abs(self.gamma - 1.0) > 1e-3:
            v = cv2.LUT(v, self.gamma_lut)

        # 5) Color/saturation balance
        if self.saturation_scale != 1.0:
            s = np.clip(s.astype(np.float32) * self.saturation_scale, 0, 255).astype(np.uint8)

        hsv = cv2.merge((h, s, v))
        balanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # 6) Optional edge-preserving smoothing to tame residual color noise
        if self.bilateral_d > 0:
            balanced = cv2.bilateralFilter(
                balanced,
                d=self.bilateral_d,
                sigmaColor=self.bilateral_sigma_color,
                sigmaSpace=self.bilateral_sigma_space,
            )

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
