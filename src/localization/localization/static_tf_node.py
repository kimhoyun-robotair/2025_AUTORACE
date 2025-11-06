#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import yaml, math
from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

def rpy_deg_to_quat(roll_deg: float, pitch_deg: float, yaw_deg: float):
    """
    Convert RPY in degrees (about parent axes) to quaternion.
    Rotation order: Roll(X) -> Pitch(Y) -> Yaw(Z).
    """
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)

    cr = math.cos(r * 0.5); sr = math.sin(r * 0.5)
    cp = math.cos(p * 0.5); sp = math.sin(p * 0.5)
    cy = math.cos(y * 0.5); sy = math.sin(y * 0.5)

    # XYZ (roll->pitch->yaw) extrinsic == ZYX intrinsic
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    return qx, qy, qz, qw

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

            if 'rotation_rpy_deg' not in item:
                raise RuntimeError(
                    f"transform {parent}->{child} must have 'rotation_rpy_deg: [roll, pitch, yaw]'"
                )
            roll_deg, pitch_deg, yaw_deg = item['rotation_rpy_deg']
            qx, qy, qz, qw = rpy_deg_to_quat(roll_deg, pitch_deg, yaw_deg)

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

            self.get_logger().info(
                f"[static TF] {parent} -> {child} | t=({tx:.3f},{ty:.3f},{tz:.3f}) m | "
                f"rpy(deg)=({roll_deg:.3f},{pitch_deg:.3f},{yaw_deg:.3f})"
            )

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

