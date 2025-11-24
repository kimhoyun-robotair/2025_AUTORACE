#!/usr/bin/env python3
'''
HSV 색상 기준으로 노란색 차선을 탐지하는 코드
'''

import rclpy
from rclpy.node import Node

import numpy as np
import cv2

from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class WhiteLaneDetector(Node):
    def __init__(self):
        super().__init__('white_lane_detector')

        # 노란색 HSV 범위 파라미터
        '''
        self.declare_parameter('hue_white_l', 15)
        self.declare_parameter('hue_white_h', 40)
        self.declare_parameter('saturation_white_l', 70)
        self.declare_parameter('saturation_white_h', 255)
        self.declare_parameter('lightness_white_l', 80)
        self.declare_parameter('lightness_white_h', 255)
        '''

        # 이건 흰색 정지선 및 횡단보도 탐지
        self.declare_parameter('hue_white_l', 0)
        self.declare_parameter('hue_white_h', 179)
        self.declare_parameter('saturation_white_l', 0)
        self.declare_parameter('saturation_white_h', 30)
        self.declare_parameter('lightness_white_l', 186)
        self.declare_parameter('lightness_white_h', 215)
        self.declare_parameter('min_aspect_ratio', 1.2)  # h/w 비율 하한 (세로가 긴 영역만)
        self.declare_parameter('min_area', 200.0)        # 너무 작은 잡음 제거
        self.declare_parameter('crop_left_ratio', 0.1)   # 좌측에서 자를 비율 (0~0.45 권장)
        self.declare_parameter('crop_right_ratio', 0.1)  # 우측에서 자를 비율 (0~0.45 권장)
        self.declare_parameter('show_crop_guides', True) # 크롭 기준선 시각화 여부

        self.update_params_from_server()

        self.bridge = CvBridge()
        self.sub = self.create_subscription(
            Image,
            '/image_balanced',          # BevNode가 퍼블리시하는 BEV 이미지
            self.image_callback,
            10
        )
        self.pub_mask = self.create_publisher(
            Image,
            '/lane/yellow_mask',    # 마스크만 퍼블리시
            10
        )

        self.counter = 0
        self.last_lane_fit = None
        self.add_on_set_parameters_callback(self.on_param_change)
        self.get_logger().info('WhiteLaneDetector initialized, subscribing to /image_bev')

    def update_params_from_server(self):
        ''' 파라미터 값 읽어오기 '''
        self.hue_white_l = self.get_parameter('hue_white_l').value
        self.hue_white_h = self.get_parameter('hue_white_h').value
        self.saturation_white_l = self.get_parameter('saturation_white_l').value
        self.saturation_white_h = self.get_parameter('saturation_white_h').value
        self.lightness_white_l = self.get_parameter('lightness_white_l').value
        self.lightness_white_h = self.get_parameter('lightness_white_h').value
        self.min_aspect_ratio = float(self.get_parameter('min_aspect_ratio').value)
        self.min_area = float(self.get_parameter('min_area').value)
        self.crop_left_ratio = float(self.get_parameter('crop_left_ratio').value)
        self.crop_right_ratio = float(self.get_parameter('crop_right_ratio').value)
        self.show_crop_guides = bool(self.get_parameter('show_crop_guides').value)

        self.crop_left_ratio = np.clip(self.crop_left_ratio, 0.0, 0.45)
        self.crop_right_ratio = np.clip(self.crop_right_ratio, 0.0, 0.45)

    def on_param_change(self, params):
        ''' 파라미터 값 변경될 경우의 콜백'''
        for p in params:
            if p.name in [
                'hue_white_l', 'hue_white_h',
                'saturation_white_l', 'saturation_white_h',
                'lightness_white_l', 'lightness_white_h',
                'min_aspect_ratio', 'min_area',
                'crop_left_ratio', 'crop_right_ratio',
                'show_crop_guides'
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

        # 1) 마스크 생성 (현재 설정은 노란색 범위)
        mask = self.mask_white_lane(img)
        mask = self.filter_vertical_regions(mask)

        # 2) 슬라이딩 윈도우 피팅은 여기서는 안 사용 (필요하면 유지 가능)
        # lane_fitx, ploty = self.fit_lane_sliding_window(mask)

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

        cv2.imshow('white_lane_debug', debug_img)
        cv2.waitKey(1)

    # ----------------- White Lane Mask -----------------

    def mask_white_lane(self, image: np.ndarray) -> np.ndarray:
        """BGR 이미지를 입력 받아 HSV로 변환 후, 지정한 HSV 범위만 마스크(0/255)로 반환."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_white = np.array([
            self.hue_white_l,
            self.saturation_white_l,
            self.lightness_white_l
        ], dtype=np.uint8)

        upper_white = np.array([
            self.hue_white_h,
            self.saturation_white_h,
            self.lightness_white_h
        ], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower_white, upper_white)
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
        """컨투어의 가로세로비와 최소 면적으로 필터링해 세로로 긴 흰색 영역만 남긴다."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered = np.zeros_like(mask)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w == 0:
                continue
            aspect = h / float(w)
            if aspect < self.min_aspect_ratio:
                continue
            cv2.drawContours(filtered, [cnt], -1, 255, thickness=-1)
        return filtered

    # ----------------- Sliding Window Lane Fitting -----------------

    def fit_lane_sliding_window(self, mask: np.ndarray):
        """
        필요하면 나중에 center_x 계산 등에 쓸 수 있도록 남겨둔 함수.
        """
        histogram = np.sum(mask[mask.shape[0] // 2:, :], axis=0)
        lane_base = int(np.argmax(histogram))

        nwindows = 20
        window_height = int(mask.shape[0] / nwindows)

        nonzero = mask.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        x_current = lane_base
        margin = 50
        minpix = 50
        lane_inds = []

        for window in range(nwindows):
            win_y_low = mask.shape[0] - (window + 1) * window_height
            win_y_high = mask.shape[0] - window * window_height
            win_x_low = x_current - margin
            win_x_high = x_current + margin

            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
            lane_inds.append(good_inds)

            if len(good_inds) > minpix:
                x_current = int(np.mean(nonzerox[good_inds]))

        if len(lane_inds) == 0:
            return None, None

        lane_inds = np.concatenate(lane_inds)

        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]

        if len(x) < 10:
            return None, None

        try:
            lane_fit = np.polyfit(y, x, 2)
            self.last_lane_fit = lane_fit
        except Exception as e:
            self.get_logger().warn(f'polyfit failed: {e}')
            if self.last_lane_fit is None:
                return None, None
            lane_fit = self.last_lane_fit

        ploty = np.linspace(0, mask.shape[0] - 1, mask.shape[0])
        lane_fitx = lane_fit[0] * ploty ** 2 + lane_fit[1] * ploty + lane_fit[2]

        return lane_fitx, ploty


def main(args=None):
    rclpy.init(args=args)
    node = WhiteLaneDetector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
