#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

class BoxController(Node):
    def __init__(self):
        super().__init__('box_controller')

        # === 파라미터 (필요하면 launch에서 override 가능) ===
        self.declare_parameter('radius', 0.95)   # [m]
        self.declare_parameter('speed', 0.5)    # [m/s] 선속도 v
        self.declare_parameter('cmd_topic', '/demo/box0_cmd_demo')
        self.declare_parameter('odom_topic', '/demo/box0_odom_demo')
        self.declare_parameter('hz', 50.0)      # 발행 주기 [Hz]

        R   = float(self.get_parameter('radius').value)
        v   = float(self.get_parameter('speed').value)
        hz  = float(self.get_parameter('hz').value)
        self.cmd_topic  = self.get_parameter('cmd_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value

        # v = R * omega
        self.omega = v / max(R, 1e-6)  # 0 나눗셈 방지
        self.v = v

        self.publisher_ = self.create_publisher(Twist, self.cmd_topic, 10)
        self.subscription_ = self.create_subscription(
            Odometry, self.odom_topic, self.listener_callback, 10
        )

        self.msg = Twist()
        self.msg.linear.x = self.v
        self.msg.angular.z = self.omega  # 반시계(+). 시계는 음수로.

        self.y = None  # 필요시 모니터링 용
        self.timer = self.create_timer(1.0 / hz, self.timer_callback)

        self.get_logger().info(
            f'Circular motion: R={R:.3f} m, v={self.v:.3f} m/s, omega={self.omega:.3f} rad/s, '
            f'publish @ {hz:.1f} Hz to {self.cmd_topic}'
        )

    def listener_callback(self, odom: Odometry):
        # 원하면 상태 로깅/모니터링에 사용
        self.y = odom.pose.pose.position.y

    def timer_callback(self):
        # 등속 원운동: v=const, omega=const
        self.publisher_.publish(self.msg)

def main(args=None):
    rclpy.init(args=args)
    node = BoxController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
