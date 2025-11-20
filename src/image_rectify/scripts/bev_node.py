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
        
        # base_link 좌표계 기준, 지면의 Z좌표 (예: -0.05m)
        #self.declare_parameter('ground_z_in_base_frame', -0.05)
        self.declare_parameter('ground_z_in_base_frame', 0.00)
        self.ground_z = self.get_parameter('ground_z_in_base_frame').get_parameter_value().double_value

        self.config_path = self.get_parameter('config_path').get_parameter_value().string_value
        self.base_frame  = self.get_parameter('base_frame').get_parameter_value().string_value
        self.camera_frame= self.get_parameter('camera_frame').get_parameter_value().string_value
        self.bev_frame   = self.get_parameter('bev_frame').get_parameter_value().string_value

        if not os.path.isfile(self.config_path):
            raise FileNotFoundError(f'config_path not found: {self.config_path}')
        cfg = yaml.safe_load(open(self.config_path, 'r'))

        # --- 1. 추가된 부분: YAML 파싱 ---
        cam_cfg = cfg['camera']
        bev_cfg = cfg['bev']
        
        # 카메라 파라미터 (Kc, D)
        self.Kc = np.array(cam_cfg['K'], dtype=np.float32).reshape((3, 3))
        self.D  = np.array(cam_cfg['D'], dtype=np.float32)
        
        # BEV 파라미터 (Kv, out_w, out_h)
        self.out_w = int(bev_cfg['output_width'])
        self.out_h = int(bev_cfg['output_height'])
        mpp = float(bev_cfg['meters_per_pixel'])
        cx_px = float(bev_cfg['cx']) # BEV 이미지 중심 X (픽셀)
        cy_px = float(bev_cfg['cy']) # BEV 이미지 중심 Y (픽셀)

        # 가상 BEV 카메라 행렬 Kv 생성
        # (X_m, Y_m) -> (u_px, v_px) 매핑
        # u = X/mpp + cx_px
        # v = Y/mpp + cy_px
        self.Kv = np.array([
            [1.0/mpp, 0,       cx_px],
            [0,       1.0/mpp, cy_px],
            [0,       0,       1]
        ], dtype=np.float32)
        # --- 1. 추가 끝 ---

        # IO
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=5)
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, '/camera/image_raw', self.image_cb, qos)
        self.pub = self.create_publisher(Image, '/image_bev', 10)

        # lazy members
        self.map1 = self.map2 = None
        self.input_size = None
        self.H = None
        
        # --- 2. 추가/수정된 부분: TF 버퍼 (타이머보다 먼저 선언!) ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.timer = self.create_timer(0.2, self.try_build_H_once)
        # --- 2. 수정 끝 ---


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

    def ensure_undistort_maps(self, w, h):
        if self.input_size == (w, h) and (self.map1 is not None or not np.any(self.D)):
            return
        self.input_size = (w, h)
        if np.any(np.abs(self.D) > 1e-12):
            # (주석 해제됨) Fisheye 모델 사용
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

        # H, out_w, out_h 모두 __init__에서 정상적으로 초기화됨
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