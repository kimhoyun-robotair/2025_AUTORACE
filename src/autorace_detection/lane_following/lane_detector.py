#!/usr/bin/env python3
'''
강한 LED 번짐이 있는 환경에서 노란색 세로 차선을 탐지하는 코드.
토픽: /image_balanced 입력 -> /lane/yellow_mask BGR 마스크 출력.
튜닝 가이드:
  - hue_yellow_*, sat_yellow_*, val_yellow_*: 노란색 범위. LED 색 섞이면 sat_yellow_l ↑, val_yellow_h ↓.
  - value_clip_percentile: 밝기 상위 퍼센타일 클리핑(LED 억제). 밝으면 90~95, 어두우면 97~100.
  - median_kernel: 번짐/노이즈 스무딩(홀수). 커질수록 선이 두꺼워짐.
  - morph_kernel/open_iter/close_iter: 잡음 제거와 단절된 선 연결. 잡음 많으면 open_iter ↑, 끊기면 close_iter ↑.
  - min_aspect_ratio/min_area/max_area: 세로선 필터. false positive가 넓게 번지면 max_area ↓.
  - crop_left_ratio/crop_right_ratio: 측면 잘라내 false positive 감소. 한쪽 차선만 필요하면 반대쪽을 0.1~0.2로 설정.
'''

import rclpy
from rclpy.node import Node

import numpy as np
import cv2

from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class YellowLaneDetector(Node):
    def __init__(self):
        super().__init__('yellow_lane_detector')

        # 색상 범위 (노란색 세로선)
        self.declare_parameter('hue_yellow_l', 15)
        self.declare_parameter('hue_yellow_h', 40)
        self.declare_parameter('sat_yellow_l', 70)
        self.declare_parameter('sat_yellow_h', 255)
        self.declare_parameter('val_yellow_l', 80)
        self.declare_parameter('val_yellow_h', 255)

        # 과다 노출/번짐 억제
        self.declare_parameter('value_clip_percentile', 95.0)  # 상위 밝기 클리핑; LED 심하면 90~95
        self.declare_parameter('median_kernel', 5)             # 번짐 스무딩; 홀수(3/5). 크면 선이 두꺼워짐

        # 형태학적 필터
        self.declare_parameter('morph_kernel', 5)              # 근처 잡음 제거 및 연결
        self.declare_parameter('open_iter', 1)                 # 작은 점 제거; 더 크게 하면 얇은 선도 사라질 수 있음
        self.declare_parameter('close_iter', 8.0)                # 끊긴 선 이어붙임; 너무 크면 덩어리화

        # 컨투어 필터
        self.declare_parameter('min_aspect_ratio', 5.0)        # h/w 비율 하한 (세로로 긴 영역만)
        self.declare_parameter('min_area', 200.0)              # 너무 작은 잡음 제거
        self.declare_parameter('max_area', 20500.0)           # 너무 큰 번짐 제거

        # 영역 크롭 (왼/오른쪽 잘라내기)
        self.declare_parameter('crop_left_ratio', 0.0)   # 좌측에서 자를 비율 (0~0.45 권장)
        self.declare_parameter('crop_right_ratio', 0.0)  # 우측에서 자를 비율 (0~0.45 권장)
        self.declare_parameter('show_crop_guides', True) # 크롭 기준선 시각화 여부

        # 밝기 자동 보정 (현재 이미지 평균 밝기가 목표보다 어두우면 살짝 밝게, 밝으면 살짝 어둡게)
        self.declare_parameter('brightness_target', 120.0)     # 목표 평균 밝기 (HSV V 채널 기준)
        self.declare_parameter('brightness_adjust_gain', 0.25) # (target-mean)/target 비율에 곱할 스케일
        self.declare_parameter('brightness_scale_min', 0.7)    # 최소 스케일 (너무 어두워지는 것 방지)
        self.declare_parameter('brightness_scale_max', 1.8)    # 최대 스케일 (너무 밝아지는 것 방지)

        # 최외곽 컨투어만 남길지 여부
        self.declare_parameter('keep_outermost_only', True)    # True면 가장 좌/우 컨투어만 유지

        self.update_params_from_server()

        self.bridge = CvBridge()
        self.sub = self.create_subscription(
            Image,
            '/image_bev',          # illumination_preprocessor 출력 (BEV)
            self.image_callback,
            10
        )
        self.pub_mask = self.create_publisher(
            Image,
            '/lane/yellow_mask',    # 마스크만 퍼블리시
            10
        )

        self.counter = 0
        self.add_on_set_parameters_callback(self.on_param_change)
        self.get_logger().info('YellowLaneDetector initialized, subscribing to /image_balanced')

    def update_params_from_server(self):
        # 파라미터 값 읽어오기
        self.hue_yellow_l = self.get_parameter('hue_yellow_l').value
        self.hue_yellow_h = self.get_parameter('hue_yellow_h').value
        self.sat_yellow_l = self.get_parameter('sat_yellow_l').value
        self.sat_yellow_h = self.get_parameter('sat_yellow_h').value
        self.val_yellow_l = self.get_parameter('val_yellow_l').value
        self.val_yellow_h = self.get_parameter('val_yellow_h').value

        self.value_clip_percentile = float(self.get_parameter('value_clip_percentile').value)
        self.median_kernel = int(self.get_parameter('median_kernel').value)

        self.morph_kernel = max(1, int(self.get_parameter('morph_kernel').value))
        if self.morph_kernel % 2 == 0:
            self.morph_kernel -= 1
        self.open_iter = int(self.get_parameter('open_iter').value)
        self.close_iter = int(self.get_parameter('close_iter').value)

        self.min_aspect_ratio = float(self.get_parameter('min_aspect_ratio').value)
        self.min_area = float(self.get_parameter('min_area').value)
        self.max_area = float(self.get_parameter('max_area').value)
        self.crop_left_ratio = float(self.get_parameter('crop_left_ratio').value)
        self.crop_right_ratio = float(self.get_parameter('crop_right_ratio').value)
        self.show_crop_guides = bool(self.get_parameter('show_crop_guides').value)
        self.brightness_target = float(self.get_parameter('brightness_target').value)
        self.brightness_adjust_gain = float(self.get_parameter('brightness_adjust_gain').value)
        self.brightness_scale_min = float(self.get_parameter('brightness_scale_min').value)
        self.brightness_scale_max = float(self.get_parameter('brightness_scale_max').value)
        self.keep_outermost_only = bool(self.get_parameter('keep_outermost_only').value)

        self.crop_left_ratio = np.clip(self.crop_left_ratio, 0.0, 0.45)
        self.crop_right_ratio = np.clip(self.crop_right_ratio, 0.0, 0.45)

    def on_param_change(self, params):
        # 파라미터 값 변경될 경우의 콜백
        for p in params:
            if p.name in [
                'hue_yellow_l', 'hue_yellow_h',
                'sat_yellow_l', 'sat_yellow_h',
                'val_yellow_l', 'val_yellow_h',
                'value_clip_percentile', 'median_kernel',
                'morph_kernel', 'open_iter', 'close_iter',
                'min_aspect_ratio', 'min_area', 'max_area',
                'crop_left_ratio', 'crop_right_ratio', 'show_crop_guides',
                'brightness_target', 'brightness_adjust_gain',
                'brightness_scale_min', 'brightness_scale_max',
                'keep_outermost_only'
            ]:
                pass
        self.update_params_from_server()
        return rclpy.parameter.SetParametersResult(successful=True)

    # ----------------- Core Callbacks -----------------

    def image_callback(self, msg: Image):
        # 프레임 드랍 (2프레임 중 1프레임만 처리)
        self.counter += 1
        if self.counter % 2 != 0:
            return

        # ROS Image -> OpenCV BGR 이미지
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        img = self.crop_sides(img)

        # 1) 노출 억제 + 색 기반 마스크
        mask = self.mask_yellow_lane(img)

        # 2) 세로 컨투어 필터링
        mask = self.filter_vertical_regions(mask)

        # 3) 마스크만 BGR로 변환해서 시각화/퍼블리시
        debug_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        if self.show_crop_guides and (self.crop_left_ratio > 0 or self.crop_right_ratio > 0):
            h, w = debug_img.shape[:2]
            # 좌/우 경계선 표시
            cv2.line(debug_img, (0, 0), (0, h - 1), (0, 255, 255), 2)
            cv2.line(debug_img, (w - 1, 0), (w - 1, h - 1), (0, 255, 255), 2)
            cv2.putText(
                debug_img,
                f'crop L:{self.crop_left_ratio:.2f} R:{self.crop_right_ratio:.2f}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

        out_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
        out_msg.header = msg.header
        self.pub_mask.publish(out_msg)

        cv2.imshow('yellow_lane_debug', debug_img)
        cv2.waitKey(1)

    # ----------------- Yellow Lane Mask -----------------

    def mask_yellow_lane(self, image: np.ndarray) -> np.ndarray:
        """BGR 입력 -> HSV 변환 -> 노란색 범위 마스크 + 형태학적 후처리."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # --- 밝기 자동 보정 시작 ---
        # 평균 밝기(mean of V)를 목표(brightness_target)에 근접시키기 위해 V 채널을 스케일링.
        # 구현: cv2.convertScaleAbs 사용. (조건: scale은 지정한 min~max 범위로 클리핑)
        mean_v = float(np.mean(v))
        if mean_v > 1e-3:
            scale = 1.0 + self.brightness_adjust_gain * ((self.brightness_target - mean_v) / max(1.0, self.brightness_target))
            scale = float(np.clip(scale, self.brightness_scale_min, self.brightness_scale_max))
            v = cv2.convertScaleAbs(v, alpha=scale, beta=0)
        # --- 밝기 자동 보정 끝 ---

        # 강한 빛 클리핑 (V 채널 상위 퍼센타일)
        clip_p = np.clip(self.value_clip_percentile, 50.0, 100.0)
        high_val = np.percentile(v, clip_p)
        if high_val > 0:
            v = np.clip(v, 0, high_val).astype(np.float32)
            v = v / max(1.0, high_val) * 255.0
        v = np.clip(v, 0, 255).astype(np.uint8)

        hsv = cv2.merge((h, s, v))

        lower = np.array([
            self.hue_yellow_l,
            self.sat_yellow_l,
            self.val_yellow_l
        ], dtype=np.uint8)

        upper = np.array([
            self.hue_yellow_h,
            self.sat_yellow_h,
            self.val_yellow_h
        ], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower, upper)

        # 스무딩으로 소금후추 노이즈 제거
        if self.median_kernel > 1:
            k = self.median_kernel
            if k % 2 == 0:
                k += 1
            mask = cv2.medianBlur(mask, k)

        # 형태학적 오프닝/클로징
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.morph_kernel, self.morph_kernel))
        if self.open_iter > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=self.open_iter)
        if self.close_iter > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=self.close_iter)

        return mask

    def crop_sides(self, image: np.ndarray) -> np.ndarray:
        """
        좌우 비율로 이미지를 잘라서 중앙 영역만 남긴다.
        crop_left_ratio=0.1, crop_right_ratio=0.1이면 좌우 10%씩 제거.
        """
        h, w = image.shape[:2]
        left_px = int(w * self.crop_left_ratio)
        right_px = int(w * self.crop_right_ratio)
        x1 = left_px
        x2 = max(x1 + 1, w - right_px)
        return image[:, x1:x2]

    def filter_vertical_regions(self, mask: np.ndarray) -> np.ndarray:
        """컨투어의 가로세로비/면적을 사용해 세로로 긴 노란 영역만 남긴다."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        kept = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            if area > self.max_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w == 0:
                continue
            aspect = h / float(w)
            if aspect < self.min_aspect_ratio:
                continue
            kept.append((cnt, x, y, w, h))

        # --- 최외곽 컨투어만 남기는 선택적 필터 시작 ---
        # keep_outermost_only가 True면 x 좌표 기준 가장 왼쪽/오른쪽 컨투어만 유지하여 중간 선을 제거한다.
        if self.keep_outermost_only and len(kept) > 2:
            kept = self._keep_outermost_components(kept)
        # --- 최외곽 컨투어만 남기는 선택적 필터 끝 ---

        filtered = np.zeros_like(mask)
        for cnt, _, _, _, _ in kept:
            cv2.drawContours(filtered, [cnt], -1, 255, thickness=-1)
        return filtered

    def _keep_outermost_components(self, contours):
        """x 좌표 기준 최외곽(가장 왼쪽/오른쪽) 컨투어만 반환한다."""
        if len(contours) <= 2:
            return contours
        left = min(contours, key=lambda c: c[1])          # 가장 왼쪽 컨투어
        right = max(contours, key=lambda c: c[1] + c[3])  # 가장 오른쪽 컨투어 (x+w 최대)
        if left is right:
            return [left]
        return [left, right]


def main(args=None):
    rclpy.init(args=args)
    node = YellowLaneDetector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
