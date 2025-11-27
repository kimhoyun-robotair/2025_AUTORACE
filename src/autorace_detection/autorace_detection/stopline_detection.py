#!/usr/bin/env python3
'''
[Horizontal Debug + Rotated Box + Yaw Display]
- 기존 로직/파라미터 100% 고정 (Final Version)
- 추가: 디버깅 화면에 Yaw 각도를 Degree 단위로 표시
'''

import rclpy
from rclpy.node import Node
import math
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class WhiteLaneHorizontalDebug(Node):
    def __init__(self):
        super().__init__('white_lane_horizontal_debug')

        # ---------------------------------------------------------
        # [파라미터 튜닝]
        # ---------------------------------------------------------
        self.declare_parameter('brightness_threshold', 100)
        self.declare_parameter('horizontal_kernel_len', 60)
        self.declare_parameter('aspect_ratio_thresh', 3.0)
        self.declare_parameter('roi_top_cut', 0.1)
        self.declare_parameter('min_area', 140)
        self.declare_parameter('crop_top', 0.75)
        self.declare_parameter('crop_bottom', 0.1)
        self.declare_parameter('crop_left', 0.3)
        self.declare_parameter('crop_right', 0.3)

        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, '/image_bev', self.image_callback, 10)
        self.pub_mask = self.create_publisher(Image, '/lane/white_mask', 10)

        self.add_on_set_parameters_callback(self.on_param_change)
        self.update_params()
        self.get_logger().info('WhiteLaneHorizontalDebug Finalized (Degree Display Added)')

    def update_params(self):
        self.thresh_val = self.get_parameter('brightness_threshold').value
        self.k_len = self.get_parameter('horizontal_kernel_len').value
        self.aspect_thresh = self.get_parameter('aspect_ratio_thresh').value
        self.roi_top = self.get_parameter('roi_top_cut').value
        self.min_area = self.get_parameter('min_area').value
        self.crop_top = self.get_parameter('crop_top').value
        self.crop_bottom = self.get_parameter('crop_bottom').value
        self.crop_left = self.get_parameter('crop_left').value
        self.crop_right = self.get_parameter('crop_right').value

    def on_param_change(self, params):
        for p in params:
            if p.name in ['brightness_threshold', 'horizontal_kernel_len', 'aspect_ratio_thresh', 'roi_top_cut', 'min_area', 'crop_top', 'crop_bottom', 'crop_left', 'crop_right']:
                pass
        self.update_params()
        return rclpy.parameter.SetParametersResult(successful=True)

    def image_callback(self, msg: Image):
        img_full = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h_full, w_full = img_full.shape[:2]

        img, crop_info = self.apply_crop(img_full)
        h, w = img.shape[:2]

        # 1. 전처리 (ROI & Gray)
        cut_h = int(h * self.roi_top)
        if cut_h > 0:
            img[0:cut_h, :] = 0

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. 이진화 & 모폴로지
        _, binary = cv2.threshold(gray, self.thresh_val, 255, cv2.THRESH_BINARY)

        kernel_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (self.k_len, 1))
        mask_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_horiz)

        kernel_restore = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        mask_final = cv2.dilate(mask_clean, kernel_restore, iterations=1)

        # 3. 필터링 & 시각화 (텍스트 추가됨)
        vis_img = img_full.copy()
        real_lane_mask = np.zeros_like(gray)

        contours, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_cnt = None
        max_valid_area = 0

        for cnt in contours:
            rect = cv2.minAreaRect(cnt)  # ((cx, cy), (w, h), angle)
            (center_x, center_y), (rect_w, rect_h), rect_angle = rect
            box_points = cv2.boxPoints(rect)
            box_points = np.int0(box_points)

            area = rect_w * rect_h
            long_side = max(rect_w, rect_h)
            short_side = min(rect_w, rect_h)
            if short_side == 0:
                continue
            aspect = long_side / short_side

            if area < self.min_area:
                continue

            info_text = f"A:{int(area)} R:{aspect:.1f}"
            cx, cy = int(center_x), int(center_y)

            if aspect >= self.aspect_thresh:
                cv2.drawContours(real_lane_mask, [cnt], -1, 255, thickness=cv2.FILLED)
                box_points_full = box_points + np.array([crop_info[0], crop_info[1]])
                cv2.drawContours(vis_img, [box_points_full], 0, (0, 255, 0), 2)
                cv2.circle(vis_img, (cx + crop_info[0], cy + crop_info[1]), 5, (0, 0, 255), -1)

                if area > max_valid_area:
                    max_valid_area = area
                    best_cnt = cnt

                cv2.putText(vis_img, info_text, (box_points_full[1][0], box_points_full[1][1]-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            else:
                box_points_full = box_points + np.array([crop_info[0], crop_info[1]])
                cv2.drawContours(vis_img, [box_points_full], 0, (0, 0, 255), 2)
                cv2.putText(vis_img, info_text, (box_points_full[1][0], box_points_full[1][1]-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Yaw 시각화(최적 정지선 기준)
        if best_cnt is not None:
            rect = cv2.minAreaRect(best_cnt)
            (cx_float, cy_float), _, _ = rect
            [vx, vy, _, _] = cv2.fitLine(best_cnt, cv2.DIST_L2, 0, 0.01, 0.01)
            line_angle = math.atan2(vy, vx)
            yaw_rad = -line_angle

            cx_int, cy_int = int(cx_float) + crop_info[0], int(cy_float) + crop_info[1]
            arrow_len = 50
            end_x = int(cx_int + arrow_len * math.sin(yaw_rad))
            end_y = int(cy_int - arrow_len * math.cos(yaw_rad))
            cv2.arrowedLine(vis_img, (cx_int, cy_int), (end_x, end_y), (0, 255, 255), 2)
            yaw_deg = math.degrees(yaw_rad)
            yaw_text = f"Yaw: {yaw_deg:.1f} deg"
            cv2.putText(vis_img, yaw_text, (cx_int - 40, cy_int + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 4. 출력
        out_img = np.zeros_like(img)
        out_img[real_lane_mask > 0] = [255, 255, 255]

        out_msg = self.bridge.cv2_to_imgmsg(out_img, encoding='bgr8')
        out_msg.header = msg.header
        self.pub_mask.publish(out_msg)

        # 디버깅 화면: 원본(크롭 보정) 위에 bbox 표시된 영상
        cv2.imshow('Debug Info View', vis_img)
        cv2.waitKey(1)

    def apply_crop(self, img: np.ndarray):
        h, w = img.shape[:2]
        top_f = np.clip(self.crop_top, 0.0, 0.95)
        bottom_f = np.clip(self.crop_bottom, 0.0, 0.95)
        left_f = np.clip(self.crop_left, 0.0, 0.95)
        right_f = np.clip(self.crop_right, 0.0, 0.95)
        if top_f + bottom_f >= 0.98 or left_f + right_f >= 0.98:
            return img, (0, 0)
        top_px = int(round(h * top_f))
        bottom_px = int(round(h * bottom_f))
        left_px = int(round(w * left_f))
        right_px = int(round(w * right_f))
        cropped = img[top_px:h - bottom_px, left_px:w - right_px]
        return cropped, (left_px, top_px)

    def _restore_coords(self, x: int, y: int, crop_info):
        """Convert cropped coords back to full-image coords."""
        left_px, top_px = crop_info
        return x + left_px, y + top_px

def main(args=None):
    rclpy.init(args=args)
    node = WhiteLaneHorizontalDebug()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
