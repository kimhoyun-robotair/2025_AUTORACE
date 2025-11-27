#!/usr/bin/env python3
'''
HSV 색상 기준으로 빨간색 / 파란색 차선을 탐지하는 코드
- /image_bev 를 구독
- 빨강/파랑 마스크를 만든 뒤 픽셀 수를 비교해서 lane color 결정
- /lane/color 에 String("red" 또는 "blue" 또는 "unknown") 퍼블리시
- /lane/color_debug 로 디버그용 컬러 이미지 퍼블리시
'''

import rclpy
from rclpy.node import Node

import numpy as np
import cv2

from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32
from cv_bridge import CvBridge


class RedBlueLaneDetector(Node):
    def __init__(self):
        super().__init__('red_blue_lane_detector')

        # 빨간색은 Hue가 0 근처를 사이에 두고 wrap-around 되기 때문에
        # 2개의 범위를 OR 해서 사용 (예: [0,10] U [170,180])
        self.declare_parameter('hue_red_l1', 0)
        self.declare_parameter('hue_red_h1', 10)
        self.declare_parameter('hue_red_l2', 170)
        self.declare_parameter('hue_red_h2', 180)
        self.declare_parameter('sat_red_l', 100)
        self.declare_parameter('sat_red_h', 255)
        self.declare_parameter('val_red_l', 100)
        self.declare_parameter('val_red_h', 255)

        # 파란색 HSV 범위 (대략적인 기본값)
        self.declare_parameter('hue_blue_l', 95)
        self.declare_parameter('hue_blue_h', 140)
        self.declare_parameter('sat_blue_l', 40)
        self.declare_parameter('sat_blue_h', 255)
        self.declare_parameter('val_blue_l', 100)
        self.declare_parameter('val_blue_h', 255)

        # 노이즈 제거용 최소 픽셀 개수
        self.declare_parameter('min_pixel_count', 1000)
        self.declare_parameter('red_dominance_ratio', 1.5)

        # 크롭 비율 (0.0~1.0). 상/하/좌/우 순으로 잘라내기.
        self.declare_parameter('crop_top', 0.8)
        self.declare_parameter('crop_bottom', 0.0)
        self.declare_parameter('crop_left', 0.4)
        self.declare_parameter('crop_right', 0.4)

        self.update_params_from_server()

        self.bridge = CvBridge()

        self.sub = self.create_subscription(
            Image,
            '/image_bev',
            self.image_callback,
            10
        )

        self.pub_color = self.create_publisher(
            String,
            '/lane/color',
            10
        )
        self.pub_center = self.create_publisher(
            Float32,
            '/lane/color_center_offset',
            10
        )

        self.pub_debug = self.create_publisher(
            Image,
            '/lane/color_debug',
            10
        )

        self.counter = 0
        self.last_color = 'unknown'

        self.get_logger().info('RedBlueLaneDetector initialized, subscribing to /image_bev')

    def update_params_from_server(self):
        '''파라미터 값 읽어오기'''
        self.hue_red_l1 = self.get_parameter('hue_red_l1').value
        self.hue_red_h1 = self.get_parameter('hue_red_h1').value
        self.hue_red_l2 = self.get_parameter('hue_red_l2').value
        self.hue_red_h2 = self.get_parameter('hue_red_h2').value
        self.sat_red_l = self.get_parameter('sat_red_l').value
        self.sat_red_h = self.get_parameter('sat_red_h').value
        self.val_red_l = self.get_parameter('val_red_l').value
        self.val_red_h = self.get_parameter('val_red_h').value

        self.hue_blue_l = self.get_parameter('hue_blue_l').value
        self.hue_blue_h = self.get_parameter('hue_blue_h').value
        self.sat_blue_l = self.get_parameter('sat_blue_l').value
        self.sat_blue_h = self.get_parameter('sat_blue_h').value
        self.val_blue_l = self.get_parameter('val_blue_l').value
        self.val_blue_h = self.get_parameter('val_blue_h').value

        self.min_pixel_count = self.get_parameter('min_pixel_count').value
        self.red_dominance_ratio = self.get_parameter('red_dominance_ratio').value

        self.crop_top = self.get_parameter('crop_top').value
        self.crop_bottom = self.get_parameter('crop_bottom').value
        self.crop_left = self.get_parameter('crop_left').value
        self.crop_right = self.get_parameter('crop_right').value

    def image_callback(self, msg: Image):
        # 프레임 드랍(2프레임 중 1프레임만 처리) – 필요 없으면 지워도 됨
        self.counter += 1
        if self.counter % 2 != 0:
            return

        # ROS Image -> OpenCV BGR
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # 크롭 적용 (비율)
        img = self.apply_crop(img)

        # HSV로 변환
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 빨강 / 파랑 마스크 생성
        mask_red = self.mask_red_lane(hsv)
        mask_blue = self.mask_blue_lane(hsv)

        # 각 색의 픽셀 개수 계산
        red_count = cv2.countNonZero(mask_red)
        blue_count = cv2.countNonZero(mask_blue)

        # 색상 결정 로직: red가 충분히 있을 때만 red, 아니면 blue, 둘 다 부족하면 unknown
        if red_count >= self.min_pixel_count and red_count >= blue_count * self.red_dominance_ratio:
            color = 'red'
            active_mask = mask_red
        elif blue_count >= self.min_pixel_count:
            color = 'blue'
            active_mask = mask_blue
        else:
            color = 'unknown'
            active_mask = None

        self.last_color = color

        # 결과 로그
        self.get_logger().info(
            f'Lane color: {color} (red={red_count}, blue={blue_count})'
        )

        # 결과 퍼블리시 (String)
        msg_color = String()
        msg_color.data = color
        self.pub_color.publish(msg_color)

        # 중심 오프셋 계산 및 퍼블리시 (mask 중심 - 이미지 중심) / width
        center_offset = 0.0
        m = None
        cx = None
        if active_mask is not None:
            m = cv2.moments(active_mask)
            if m["m00"] > 1e-3:
                cx = m["m10"] / m["m00"]
                width = active_mask.shape[1]
                center_offset = float((cx - width / 2.0) / width)
        self.pub_center.publish(Float32(data=center_offset))

        # 디버그용 이미지 생성: 빨간 마스크는 빨간채널, 파란 마스크는 파란채널
        debug_img = np.zeros_like(img)
        debug_img[:, :, 2] = mask_red        # R 채널
        debug_img[:, :, 0] = mask_blue       # B 채널

        # 중심 마크와 텍스트 overlay
        h, w, _ = debug_img.shape
        cx_img = int(w / 2)
        cv2.line(debug_img, (cx_img, 0), (cx_img, h), (255, 255, 0), 1)  # 이미지 중앙선
        if active_mask is not None and m["m00"] > 1e-3:
            cx_mask = int(cx)
            cy_mask = int(m["m01"] / m["m00"])
            cv2.circle(debug_img, (cx_mask, cy_mask), 6, (0, 255, 0), 2)
            cv2.line(debug_img, (cx_img, cy_mask), (cx_mask, cy_mask), (0, 255, 255), 2)

        # 이미지 크기에 따라 텍스트 크기/위치 자동 조정
        base = min(h, w)
        font_scale = max(0.5, min(1.5, base / 400.0))
        thickness = 1 if base < 400 else 2
        text_y1 = int(0.1 * h)
        text_y2 = int(0.18 * h)

        cv2.putText(
            debug_img,
            f'{color}',
            (int(0.06 * w), text_y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )
        #cv2.putText(
        #    debug_img,
        #    f'{center_offset:.3f}',
        #    (int(0.03 * w), text_y2),
        #    cv2.FONT_HERSHEY_SIMPLEX,
        #    max(0.5, font_scale * 0.8),
        #    (200, 200, 200),
        #    thickness,
        #    cv2.LINE_AA
        #)

        # ROS Image로 퍼블리시
        out_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
        out_msg.header = msg.header
        self.pub_debug.publish(out_msg)

        # 필요하면 imshow로도 확인
        cv2.imshow('lane_color_debug', debug_img)
        cv2.waitKey(1)

    def mask_red_lane(self, hsv: np.ndarray) -> np.ndarray:
        """
        빨간색 Hue는 0 근처 wrap-around 때문에
        [hue_red_l1, hue_red_h1] U [hue_red_l2, hue_red_h2] 두 범위를 합쳐서 사용.
        """
        lower_red_1 = np.array(
            [self.hue_red_l1, self.sat_red_l, self.val_red_l],
            dtype=np.uint8
        )
        upper_red_1 = np.array(
            [self.hue_red_h1, self.sat_red_h, self.val_red_h],
            dtype=np.uint8
        )

        lower_red_2 = np.array(
            [self.hue_red_l2, self.sat_red_l, self.val_red_l],
            dtype=np.uint8
        )
        upper_red_2 = np.array(
            [self.hue_red_h2, self.sat_red_h, self.val_red_h],
            dtype=np.uint8
        )

        mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
        mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
        mask = cv2.bitwise_or(mask1, mask2)

        return mask

    def mask_blue_lane(self, hsv: np.ndarray) -> np.ndarray:
        """
        파란색 Hue는 하나의 구간으로 충분.
        """
        lower_blue = np.array(
            [self.hue_blue_l, self.sat_blue_l, self.val_blue_l],
            dtype=np.uint8
        )
        upper_blue = np.array(
            [self.hue_blue_h, self.sat_blue_h, self.val_blue_h],
            dtype=np.uint8
        )

        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        return mask

    def apply_crop(self, img: np.ndarray) -> np.ndarray:
        """
        비율 기반 크롭. 값이 너무 커서 영상이 사라지면 크롭을 스킵.
        """
        h, w = img.shape[:2]
        top_f = min(max(self.crop_top, 0.0), 0.95)
        bottom_f = min(max(self.crop_bottom, 0.0), 0.95)
        left_f = min(max(self.crop_left, 0.0), 0.95)
        right_f = min(max(self.crop_right, 0.0), 0.95)

        if top_f + bottom_f >= 0.98 or left_f + right_f >= 0.98:
            self.get_logger().warn('Crop fractions too large; skipping crop this frame.')
            return img

        top_px = int(round(top_f * h))
        bottom_px = int(round(bottom_f * h))
        left_px = int(round(left_f * w))
        right_px = int(round(right_f * w))

        return img[top_px:h - bottom_px, left_px:w - right_px]


def main(args=None):
    rclpy.init(args=args)
    node = RedBlueLaneDetector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
