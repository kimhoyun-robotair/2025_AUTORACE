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
        self.declare_parameter('value_scale', 0.55)          # <1.0 darkens highlights
        self.declare_parameter('saturation_scale', 1.0)      # Optional color boost
        self.declare_parameter('use_clahe', True)            # Contrast Limited AHE on V channel
        self.declare_parameter('clahe_clip_limit', 2.0)
        self.declare_parameter('clahe_tile_grid_size', 12)
        self.declare_parameter('use_gamma', True)           # Gamma correction on V channel
        self.declare_parameter('gamma', 4.0)                 # >1 darkens, <1 brightens here
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
        """
        기존 전처리 + LAB 색상 공간을 활용한 노란색 강조 및 난반사 제거
        """
        work = image.copy()

        # 1. [선택] 가우시안 블러 (노이즈 제거)
        if self.blur_kernel_size > 1:
            work = cv2.GaussianBlur(work, (self.blur_kernel_size, self.blur_kernel_size), 0)

        # ---------------------------------------------------------
        # 전략 A: 고휘도 Glare(순수 흰색 난반사) 마스킹
        # ---------------------------------------------------------
        # 그레이스케일 변환 후 아주 밝은 부분(240~255)은 색상 정보가 없으므로 
        # 노란색 검출에 방해만 됩니다. 이를 미리 검정색으로 눌러줍니다.
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        # 임계값은 환경에 따라 230~250 사이 조절
        _, glare_mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)
        
        # 원본에서 눈부신 부분을 약간 어둡게 하거나 inpainting 할 수 있지만,
        # 여기서는 단순히 해당 영역의 채도를 죽이거나 값을 낮추는 방식을 씁니다.
        # (단순하게는 해당 영역을 검정으로 칠해버리는게 Lane 검출엔 유리할 수 있습니다)
        work[glare_mask > 0] = [0, 0, 0] 

        # ---------------------------------------------------------
        # 전략 B: LAB 색상 공간 활용 (노란색 강조의 핵심)
        # ---------------------------------------------------------
        lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # LAB에서 b-channel은 [파랑(낮음) <-> 노랑(높음)] 정보를 가집니다.
        # 노란색 차선은 b_channel 값이 매우 높게 나타납니다.
        # 조명이 강해도 b값은 흰색 조명(중립)과 노란색(높음)이 잘 분리됩니다.
        
        # CLAHE를 L(Lightness) 채널에만 적용하여 디테일을 살립니다.
        if self.use_clahe:
            clahe = cv2.createCLAHE(
                clipLimit=max(0.1, self.clahe_clip_limit),
                tileGridSize=(self.clahe_tile_grid_size, self.clahe_tile_grid_size)
            )
            l_channel = clahe.apply(l_channel)

        # 여기서 b_channel을 강조하는 것이 중요합니다.
        # 노란색 성분(b)을 스케일링하여 더욱 도드라지게 만듭니다.
        # b_channel 값 128이 무채색(0)입니다. 128 이상이 노란색 계열입니다.
        # 128 이상인 부분에 가중치를 줍니다.
        yellow_mask = (b_channel > 140).astype(np.uint8) # 140은 튜닝 필요
        
        # 시각적 확인을 위해 다시 합칩니다. 
        # 실제 Lane Detection 로직에서는 b_channel만 써도 무방합니다.
        lab_merged = cv2.merge((l_channel, a_channel, b_channel))
        balanced_bgr = cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)

        # ---------------------------------------------------------
        # 전략 C: 형태학적 필터링 (Morphology Top-Hat) - 옵션
        # ---------------------------------------------------------
        # 차선 두께에 맞는 커널 사이즈 설정 (BEV 이미지 픽셀 기준)
        # 예: 차선 폭이 약 10~20픽셀이라면
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        # tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        # 이 tophat 이미지를 mask로 활용할 수도 있습니다.

        # ---------------------------------------------------------
        # 최종 결과 반환
        # ---------------------------------------------------------
        # 디버깅을 위해 b_channel을 시각화 해보시는 것을 강력 추천합니다.
        if self.show_debug:
            # b_channel이 노란색을 얼마나 잘 잡는지 확인
            cv2.imshow('DEBUG_B_CHANNEL', b_channel)
            cv2.imshow('DEBUG_GLARE_MASK', glare_mask)

        return balanced_bgr

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