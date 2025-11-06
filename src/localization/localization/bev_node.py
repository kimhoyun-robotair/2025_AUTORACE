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

        self.config_path = self.get_parameter('config_path').get_parameter_value().string_value
        self.base_frame  = self.get_parameter('base_frame').get_parameter_value().string_value
        self.camera_frame= self.get_parameter('camera_frame').get_parameter_value().string_value
        self.bev_frame   = self.get_parameter('bev_frame').get_parameter_value().string_value

        if not os.path.isfile(self.config_path):
            raise FileNotFoundError(f'config_path not found: {self.config_path}')
        cfg = yaml.safe_load(open(self.config_path, 'r'))

        # intrinsics
        self.Kc = np.array(cfg['camera']['K'], float).reshape(3,3)
        self.D  = np.array(cfg['camera'].get('D', [0,0,0,0]), float).reshape(-1,)

        # BEV output & scale
        bev_cfg = cfg['bev']
        self.out_w = int(bev_cfg['output_width'])
        self.out_h = int(bev_cfg['output_height'])
        s = float(bev_cfg['meters_per_pixel'])  # m/px
        cx_v = bev_cfg.get('cx', self.out_w/2.0)
        cy_v = bev_cfg.get('cy', self.out_h/2.0)
        self.Kv = np.array([[1.0/s, 0,     cx_v],
                            [0,     1.0/s, cy_v],
                            [0,     0,     1   ]], float)

        # TF
        self.tf_buffer = Buffer(); self.tf_listener = TransformListener(self.tf_buffer, self)

        # IO
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=5)
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, '/camera/image_raw', self.image_cb, qos)
        self.pub = self.create_publisher(Image, '/camera/image_bev', 10)

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
        tf_base_to_cam = self.lookup(self.camera_frame, self.base_frame)
        tf_base_to_bev = self.lookup(self.bev_frame, self.base_frame)
        if tf_base_to_cam is None or tf_base_to_bev is None:
            return

        R_CB, t_CB = transform_to_R_t(tf_base_to_cam)
        R_VB, t_VB = transform_to_R_t(tf_base_to_bev)
        AC = np.concatenate([R_CB[:,[0]], R_CB[:,[1]], t_CB], axis=1)
        AV = np.concatenate([R_VB[:,[0]], R_VB[:,[1]], t_VB], axis=1)

        try:
            AC_inv = np.linalg.inv(AC)
            Kc_inv = np.linalg.inv(self.Kc)
        except np.linalg.LinAlgError:
            self.get_logger().error('Singular matrix while inverting AC/Kc. Check TF & intrinsics.')
            return

        self.H = self.Kv @ AV @ AC_inv @ Kc_inv
        self.get_logger().info(f'H initialized:\n{self.H}')
        self.timer.cancel()

    def ensure_undistort_maps(self, w, h):
        if self.input_size == (w, h) and (self.map1 is not None or not np.any(self.D)):
            return
        self.input_size = (w, h)
        if np.any(np.abs(self.D) > 1e-12):
            self.map1, self.map2 = cv2.initUndistortRectifyMap(
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

