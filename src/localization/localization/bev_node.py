#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2, yaml, os
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
from builtin_interfaces.msg import Time

def transform_to_R_t(tf: TransformStamped):
    q = tf.transform.rotation; tmsg = tf.transform.translation
    x,y,z,w = q.x,q.y,q.z,q.w
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
    ], float)
    t = np.array([[tmsg.x],[tmsg.y],[tmsg.z]], float)
    return R, t

class BevNode(Node):
    def __init__(self):
        super().__init__('bev_node')

        # params
        self.declare_parameter('config_path', '')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('camera_frame', 'camera_optical_frame')
        self.declare_parameter('bev_frame', 'bev')
        
        # --- 추가된 부분: 지면 높이 파라미터 ---
        # base_link 좌표계 기준, 지면의 Z좌표 (예: -0.7m)
        self.declare_parameter('ground_z_in_base_frame', -0.7)
        self.ground_z = self.get_parameter('ground_z_in_base_frame').get_parameter_value().double_value
        # --- 추가 끝 ---

        self.config_path = self.get_parameter('config_path').get_parameter_value().string_value
        self.base_frame  = self.get_parameter('base_frame').get_parameter_value().string_value
        self.camera_frame= self.get_parameter('camera_frame').get_parameter_value().string_value
        self.bev_frame   = self.get_parameter('bev_frame').get_parameter_value().string_value

        if not os.path.isfile(self.config_path):
            raise FileNotFoundError(f'config_path not found: {self.config_path}')
        cfg = yaml.safe_load(open(self.config_path, 'r'))

        # ... (Kv, TF, IO 관련 코드는 동일) ...
        # IO
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=5)
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, '/image_raw', self.image_cb, qos)
        self.pub = self.create_publisher(Image, '/image_bev', 10)

        # lazy members
        self.map1 = self.map2 = None
        self.input_size = None
        self.H = None
        self.timer = self.create_timer(0.2, self.try_build_H_once)


    def lookup(self, target_frame, source_frame, stamp: Time=None):
        try:
            return self.tf_buffer.lookup_transform(
                target_frame, source_frame, rclpy.time.Time() if stamp is None else stamp
            )
        except Exception as e:
            self.get_logger().warn(f'TF {source_frame}->{target_frame} not ready: {e}')
            return None

    def try_build_H_once(self):
        if self.H is not None:
            return
        
        # 1. TF 조회
        tf_base_to_cam = self.lookup(self.camera_frame, self.base_frame)
        tf_base_to_bev = self.lookup(self.bev_frame, self.base_frame)
        if tf_base_to_cam is None or tf_base_to_bev is None:
            return

        # 2. R, t 추출
        R_CB, t_CB = transform_to_R_t(tf_base_to_cam) # base -> camera
        R_VB, t_VB = transform_to_R_t(tf_base_to_bev) # base -> bev
        h = self.ground_z # 지면 높이 (스칼라)

        # 3. H_ground_to_image (H_g2i) 계산 (R의 모든 열 사용)
        r1_C = R_CB[:, [0]]
        r2_C = R_CB[:, [1]]
        r3_C = R_CB[:, [2]]
        H_g2i = self.Kc @ np.concatenate([r1_C, r2_C, t_CB + h * r3_C], axis=1)

        # 4. H_ground_to_bev (H_g2b) 계산 (R의 모든 열 사용)
        r1_V = R_VB[:, [0]]
        r2_V = R_VB[:, [1]]
        r3_V = R_VB[:, [2]]
        # (참고: bev_frame이 base_link와 Z축이 같다면 r3_V는 [0,0,k] 형태일 것임)
        H_g2b = self.Kv @ np.concatenate([r1_V, r2_V, t_VB + h * r3_V], axis=1)

        # 5. 최종 H = H_g2b * (H_g2i)^-1 계산
        try:
            H_g2i_inv = np.linalg.inv(H_g2i)
        except np.linalg.LinAlgError:
            self.get_logger().error('Singular matrix H_g2i. Check TF & intrinsics.')
            return

        self.H = H_g2b @ H_g2i_inv
        self.get_logger().info(f'H initialized (Ground Z={h:.3f}m):\n{self.H}')
        self.timer.cancel()

    # ... (ensure_undistort_maps, image_cb, main 함수는 동일) ...
    def ensure_undistort_maps(self, w, h):
        if self.input_size == (w, h) and (self.map1 is not None or not np.any(self.D)):
            return
        self.input_size = (w, h)
        if np.any(np.abs(self.D) > 1e-12):
            # self.get_logger().info('Using cv2.initUndistortRectifyMap (standard lens model)')
            # self.map1, self.map2 = cv2.initUndistortRectifyMap(
            #     self.Kc, self.D, np.eye(3), self.Kc, (w, h), cv2.CV_16SC2
            # )
            # 만약 fisheye 모델을 사용해야 한다면, 아래 코드를 대신 사용하세요.
            self.get_logger().info('Using cv2.fisheye.initUndistortRectifyMap (fisheye lens model)')
            self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
                self.Kc, self.D, np.eye(3), self.Kc, (w, h), cv2.CV_16SC2
            )
            self.get_logger().info(f'Undistort maps created for input {w}x{h}.')
        else:
            self.map1 = self.map2 = None
            self.get_logger().info('No distortion coefficients; skipping undistort.')

    def image_cb(self, msg: Image):
        self.ensure_undistort_maps(msg.width, msg.height)

        if self.H is None:
            self.try_build_H_once()
            if self.H is None:
                return

        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if self.map1 is not None:
            img = cv2.remap(img, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)

        bev = cv2.warpPerspective(img, self.H, (self.out_w, self.out_h), flags=cv2.INTER_LINEAR)
        out = self.bridge.cv2_to_imgmsg(bev, encoding='bgr8')
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.bev_frame
        self.pub.publish(out)

def main():
    rclpy.init()
    node = BevNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
