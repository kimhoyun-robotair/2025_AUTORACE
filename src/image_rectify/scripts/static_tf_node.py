#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import yaml  # math 라이브러리 제거
from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

# rpy_deg_to_quat 함수 전체 삭제

class StaticTFNode(Node):
    def __init__(self):
        super().__init__('static_tf_node')
        self.declare_parameter('tf_config_path', '')
        path = self.get_parameter('tf_config_path').get_parameter_value().string_value
        if not path:
            raise RuntimeError('tf_config_path parameter is required.')

        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)

        self.broadcaster = StaticTransformBroadcaster(self)
        transforms = []

        for item in cfg.get('transforms', []):
            parent = item['parent']
            child  = item['child']
            tx, ty, tz = item.get('translation', [0.0, 0.0, 0.0])

            # --- 수정된 부분: RPY 대신 쿼터니언 직접 읽기 ---
            if 'rotation_quat' not in item:
                raise RuntimeError(
                    f"transform {parent}->{child} must have 'rotation_quat: [x, y, z, w]'"
                )
            
            # RPY 관련 코드 삭제
            qx, qy, qz, qw = item['rotation_quat']
            # --- 수정 끝 ---

            tf = TransformStamped()
            tf.header.frame_id = parent
            tf.child_frame_id = child
            tf.transform.translation.x = float(tx)
            tf.transform.translation.y = float(ty)
            tf.transform.translation.z = float(tz)
            tf.transform.rotation.x = float(qx)
            tf.transform.rotation.y = float(qy)
            tf.transform.rotation.z = float(qz)
            tf.transform.rotation.w = float(qw)
            transforms.append(tf)

            # --- 수정된 부분: 로그 메시지 변경 ---
            self.get_logger().info(
                f"[static TF] {parent} -> {child} | t=({tx:.3f},{ty:.3f},{tz:.3f}) m | "
                f"q=({qx:.4f},{qy:.4f},{qz:.4f},{qw:.4f})"
            )
            # --- 수정 끝 ---

        self.broadcaster.sendTransform(transforms)
        self.get_logger().info(f'Published {len(transforms)} static transforms.')

def main():
    rclpy.init()
    node = StaticTFNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()