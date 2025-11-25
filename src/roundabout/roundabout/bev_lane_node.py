#!/usr/bin/env python3
import math
from typing import List, Tuple

import rclpy
from rclpy.node import Node

import numpy as np
import cv2
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose2D


class BevLanePoseNode(Node):
    def __init__(self):
        super().__init__('bev_lane_node')

        # ====== 파라미터 ======
        self.declare_parameters('', [
            ('image_topic', '/image_bev'),
            ('output_topic', '/lane_pose'),
            ('px_per_meter', 200.0),
            ('roi_y_min', 50),
            ('roi_y_max', 250),
            ('min_edge_pixels', 300),
            ('canny_threshold1', 50),
            ('canny_threshold2', 150),
            ('sample_row_step', 5),
        ])

        p = self.get_parameter
        self.image_topic = p('image_topic').value
        self.output_topic = p('output_topic').value
        self.px_per_meter = float(p('px_per_meter').value)
        self.roi_y_min = int(p('roi_y_min').value)
        self.roi_y_max = int(p('roi_y_max').value)
        self.min_edge_pixels = int(p('min_edge_pixels').value)
        self.canny_th1 = int(p('canny_threshold1').value)
        self.canny_th2 = int(p('canny_threshold2').value)
        self.sample_row_step = int(p('sample_row_step').value)

        # ====== ROS I/O ======
        self.bridge = CvBridge()

        self.sub_img = self.create_subscription(
            Image, self.image_topic, self.image_callback, 10
        )
        self.pub_pose = self.create_publisher(
            Pose2D, self.output_topic, 10
        )

        # 마지막 유효 값 (차선 잠깐 안 보일 때 fallback용)
        self.last_pose = Pose2D()
        self.have_last_pose = False

        self.get_logger().info(
            f'BevLanePoseNode started. Subscribing: {self.image_topic}, '
            f'Publishing: {self.output_topic}'
        )

    # ==========================
    # Image callback
    # ==========================
    def image_callback(self, msg: Image):
        # ROS Image → OpenCV BGR
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge conversion failed: {e}')
            return

        ok, lateral, heading = self.process_bev_image(cv_image)

        if ok:
            pose = Pose2D()
            pose.x = float(lateral)       # lateral offset [m], 왼쪽 +
            pose.y = 0.0
            pose.theta = float(heading)   # heading error [rad], CCW +

            self.pub_pose.publish(pose)
            self.last_pose = pose
            self.have_last_pose = True
        else:
            # 차선을 못 찾으면 마지막 pose 유지 (또는 publish 안 해도 됨)
            if self.have_last_pose:
                self.pub_pose.publish(self.last_pose)

    # ==========================
    # 핵심: BEV 이미지 처리 (Canny 기반)
    # ==========================
    def process_bev_image(self, img_bgr) -> Tuple[bool, float, float]:
        """
        BEV BGR 이미지를 받아서:
          - grayscale + Canny로 노란 차선의 edge 검출
          - ROI 내 여러 row에서 edge 분포를 보고 차선 폭/중앙 계산
          - 중앙 line을 직선으로 근사하여 heading + lateral offset 계산
        반환: (ok, lateral[m], heading[rad])
        """

        h, w, _ = img_bgr.shape

        # ROI 보정
        y_min = max(0, min(self.roi_y_min, h - 1))
        y_max = max(0, min(self.roi_y_max, h - 1))
        if y_min >= y_max:
            y_min = 0
            y_max = h - 1

        roi = img_bgr[y_min:y_max, :]

        # BGR → GRAY
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 살짝 블러 (노이즈 줄이기)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny edge
        edges = cv2.Canny(gray_blur, self.canny_th1, self.canny_th2)

        # edge 개수 체크
        edge_count = cv2.countNonZero(edges)
        if edge_count < self.min_edge_pixels:
            return (False, 0.0, 0.0)

        # 여러 row에서 edge 분포를 보고 "왼쪽 boundary ~ 오른쪽 boundary" 중앙을 샘플링
        points_y: List[int] = []
        points_x: List[float] = []

        for local_row in range(0, edges.shape[0], self.sample_row_step):
            row = edges[local_row, :]
            xs = np.where(row > 0)[0]  # edge가 있는 x 인덱스들
            if xs.size == 0:
                continue

            # 가장 왼쪽 / 가장 오른쪽 edge
            left_x = float(xs.min())
            right_x = float(xs.max())

            center_x = 0.5 * (left_x + right_x)
            global_y = y_min + local_row

            points_x.append(center_x)
            points_y.append(global_y)

        if len(points_x) < 5:
            # 유효한 샘플 너무 적으면 실패로 본다
            return (False, 0.0, 0.0)

        ys = np.array(points_y, dtype=np.float32)
        xs = np.array(points_x, dtype=np.float32)

        # x = a*y + b 로 직선 피팅
        a, b = np.polyfit(ys, xs, 1)

        # ====== heading error ======
        # 이미지 좌표 (BEV):
        #   - x: 오른쪽 +
        #   - y: 아래 +
        #
        # 라인이 y 증가에 따라 x가 어떻게 바뀌는지를 나타내는 기울기 a.
        # a = 0 이면 라인이 완전 수직 (차의 진행 방향과 정렬) → heading_error ≈ 0
        # a > 0 이면 y가 증가할수록 x가 증가 → 오른쪽으로 기울어진 선
        # heading = atan(a) 로 대략적인 yaw 편차를 표현 (튜닝 필요)
        heading = math.atan(a)

        # ====== lateral offset ======
        # "가장 가까운 y"에서의 중앙 x와 이미지 중심 x 차이를 이용
        y_near = y_max
        x_on_line = a * y_near + b

        img_center_x = w / 2.0
        offset_px = img_center_x - x_on_line  # 왼쪽 +, 오른쪽 -

        lateral_m = offset_px / self.px_per_meter

        return (True, lateral_m, heading)


def main(args=None):
    rclpy.init(args=args)
    node = BevLanePoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
import math
from typing import List, Tuple

import rclpy
from rclpy.node import Node

import numpy as np
import cv2
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose2D


class BevLanePoseNode(Node):
    def __init__(self):
        super().__init__('bev_lane_node')

        # ====== 파라미터 ======
        self.declare_parameters('', [
            ('image_topic', '/bev_image'),
            ('output_topic', '/lane_pose'),
            ('px_per_meter', 200.0),
            ('roi_y_min', 50),
            ('roi_y_max', 250),
            ('min_edge_pixels', 300),
            ('canny_threshold1', 50),
            ('canny_threshold2', 150),
            ('sample_row_step', 5),
        ])

        p = self.get_parameter
        self.image_topic = p('image_topic').value
        self.output_topic = p('output_topic').value
        self.px_per_meter = float(p('px_per_meter').value)
        self.roi_y_min = int(p('roi_y_min').value)
        self.roi_y_max = int(p('roi_y_max').value)
        self.min_edge_pixels = int(p('min_edge_pixels').value)
        self.canny_th1 = int(p('canny_threshold1').value)
        self.canny_th2 = int(p('canny_threshold2').value)
        self.sample_row_step = int(p('sample_row_step').value)

        # ====== ROS I/O ======
        self.bridge = CvBridge()

        self.sub_img = self.create_subscription(
            Image, self.image_topic, self.image_callback, 10
        )
        self.pub_pose = self.create_publisher(
            Pose2D, self.output_topic, 10
        )

        # 마지막 유효 값 (차선 잠깐 안 보일 때 fallback용)
        self.last_pose = Pose2D()
        self.have_last_pose = False

        self.get_logger().info(
            f'BevLanePoseNode started. Subscribing: {self.image_topic}, '
            f'Publishing: {self.output_topic}'
        )

    # ==========================
    # Image callback
    # ==========================
    def image_callback(self, msg: Image):
        # ROS Image → OpenCV BGR
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge conversion failed: {e}')
            return

        ok, lateral, heading = self.process_bev_image(cv_image)

        if ok:
            pose = Pose2D()
            pose.x = float(lateral)       # lateral offset [m], 왼쪽 +
            pose.y = 0.0
            pose.theta = float(heading)   # heading error [rad], CCW +

            self.pub_pose.publish(pose)
            self.last_pose = pose
            self.have_last_pose = True
        else:
            # 차선을 못 찾으면 마지막 pose 유지 (또는 publish 안 해도 됨)
            if self.have_last_pose:
                self.pub_pose.publish(self.last_pose)

    # ==========================
    # 핵심: BEV 이미지 처리 (Canny 기반)
    # ==========================
    def process_bev_image(self, img_bgr) -> Tuple[bool, float, float]:
        """
        BEV BGR 이미지를 받아서:
          - grayscale + Canny로 노란 차선의 edge 검출
          - ROI 내 여러 row에서 edge 분포를 보고 차선 폭/중앙 계산
          - 중앙 line을 직선으로 근사하여 heading + lateral offset 계산
        반환: (ok, lateral[m], heading[rad])
        """

        h, w, _ = img_bgr.shape

        # ROI 보정
        y_min = max(0, min(self.roi_y_min, h - 1))
        y_max = max(0, min(self.roi_y_max, h - 1))
        if y_min >= y_max:
            y_min = 0
            y_max = h - 1

        roi = img_bgr[y_min:y_max, :]

        # BGR → GRAY
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 살짝 블러 (노이즈 줄이기)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny edge
        edges = cv2.Canny(gray_blur, self.canny_th1, self.canny_th2)

        # edge 개수 체크
        edge_count = cv2.countNonZero(edges)
        if edge_count < self.min_edge_pixels:
            return (False, 0.0, 0.0)

        # 여러 row에서 edge 분포를 보고 "왼쪽 boundary ~ 오른쪽 boundary" 중앙을 샘플링
        points_y: List[int] = []
        points_x: List[float] = []

        for local_row in range(0, edges.shape[0], self.sample_row_step):
            row = edges[local_row, :]
            xs = np.where(row > 0)[0]  # edge가 있는 x 인덱스들
            if xs.size == 0:
                continue

            # 가장 왼쪽 / 가장 오른쪽 edge
            left_x = float(xs.min())
            right_x = float(xs.max())

            center_x = 0.5 * (left_x + right_x)
            global_y = y_min + local_row

            points_x.append(center_x)
            points_y.append(global_y)

        if len(points_x) < 5:
            # 유효한 샘플 너무 적으면 실패로 본다
            return (False, 0.0, 0.0)

        ys = np.array(points_y, dtype=np.float32)
        xs = np.array(points_x, dtype=np.float32)

        # x = a*y + b 로 직선 피팅
        a, b = np.polyfit(ys, xs, 1)

        # ====== heading error ======
        # 이미지 좌표 (BEV):
        #   - x: 오른쪽 +
        #   - y: 아래 +
        #
        # 라인이 y 증가에 따라 x가 어떻게 바뀌는지를 나타내는 기울기 a.
        # a = 0 이면 라인이 완전 수직 (차의 진행 방향과 정렬) → heading_error ≈ 0
        # a > 0 이면 y가 증가할수록 x가 증가 → 오른쪽으로 기울어진 선
        # heading = atan(a) 로 대략적인 yaw 편차를 표현 (튜닝 필요)
        heading = math.atan(a)

        # ====== lateral offset ======
        # "가장 가까운 y"에서의 중앙 x와 이미지 중심 x 차이를 이용
        y_near = y_max
        x_on_line = a * y_near + b

        img_center_x = w / 2.0
        offset_px = img_center_x - x_on_line  # 왼쪽 +, 오른쪽 -

        lateral_m = offset_px / self.px_per_meter

        return (True, lateral_m, heading)


def main(args=None):
    rclpy.init(args=args)
    node = BevLanePoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

