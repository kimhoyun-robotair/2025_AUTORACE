#!/usr/bin/env python3
'''
HSV 색상 기준으로 흰색 정지선 및 횡단보도를 탐지하는 코드
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

        # 이건 흰색 정지선 및 횡단보도 탐지
        self.declare_parameter('hue_white_l', 0)
        self.declare_parameter('hue_white_h', 179)
        self.declare_parameter('saturation_white_l', 0)
        self.declare_parameter('saturation_white_h', 70)
        self.declare_parameter('lightness_white_l', 105)
        self.declare_parameter('lightness_white_h', 255)

        # 파라미터 값 읽기
        self.update_params_from_server()

        # 이미지 입출력
        self.bridge = CvBridge()
        self.sub = self.create_subscription(
            Image,
            '/image_bev',          # BevNode가 퍼블리시하는 BEV 이미지
            self.image_callback,
            10
        )
        self.pub_mask = self.create_publisher(
            Image,
            '/lane/white_mask',    # 마스크만 퍼블리시
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

    def on_param_change(self, params):
        ''' 파라미터 값 변경 콜백 함수 '''
        for p in params:
            if p.name in [
                'hue_white_l', 'hue_white_h',
                'saturation_white_l', 'saturation_white_h',
                'lightness_white_l', 'lightness_white_h'
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

        # 1) 마스크 생성 (현재 설정은 흰색 범위)
        mask = self.mask_white_lane(img)

        # 1-1) 가로선 강조: 컨투어의 가로/세로 비가 큰 것만 남김
        mask = self.keep_horizontal(mask)

        # 2) 슬라이딩 윈도우 피팅은 여기서는 안 사용 (필요하면 유지 가능)
        # lane_fitx, ploty = self.fit_lane_sliding_window(mask)

        # 3) 마스크만 BGR로 변환해서 시각화/퍼블리시
        debug_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

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

    def keep_horizontal(self, mask: np.ndarray) -> np.ndarray:
        """컨투어 중 가로로 긴 것(가로/세로 비율이 threshold 이상)만 남김."""
        aspect_threshold = 3.0  # 가로가 세로보다 3배 이상일 때만 허용
        min_area = 500.0        # 너무 작은 잡음 제거

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered = np.zeros_like(mask)
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if h == 0:
                continue
            aspect = w / float(h)
            area = w * h
            if aspect >= aspect_threshold and area >= min_area:
                cv2.drawContours(filtered, [c], -1, 255, thickness=cv2.FILLED)
        return filtered

    # ----------------- Sliding Window Lane Fitting -----------------

    def fit_lane_sliding_window(self, mask: np.ndarray):
        """
        필요하면 나중에 center_x 계산 등에 쓸 수 있도록 남겨둔 함수.
        지금은 시각화에는 사용하지 않는다.
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
