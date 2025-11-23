#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_srvs.srv import SetBool
from rclpy.duration import Duration

class GateBoxController(Node):
    def __init__(self):
        super().__init__('gate_box_controller')

        # === Parameters ===
        self.declare_parameter('cmd_topic', '/demo/cross_cmd_demo')  # planar_move remap와 일치
        self.declare_parameter('hz', 50.0)       # publish rate [Hz]
        self.declare_parameter('move_speed_y', 0.5)  # [m/s] y축 속도
        self.declare_parameter('move_duration', 2.0) # [s]  유지 시간

        self.cmd_topic = self.get_parameter('cmd_topic').value
        self.hz = float(self.get_parameter('hz').value)
        self.move_speed_y = float(self.get_parameter('move_speed_y').value)
        self.move_duration = float(self.get_parameter('move_duration').value)

        # Publisher
        self.pub = self.create_publisher(Twist, self.cmd_topic, 50)

        # State
        self.active_until = None   # rclpy.time.Time or None
        self.active = False

        # Service: True면 동작 시작
        self.srv = self.create_service(SetBool, 'open_crossing',
                                       self.handle_open_crossing)

        # Timer
        self.timer = self.create_timer(1.0 / self.hz, self.on_timer)

        self.get_logger().info(
            f'GateBoxController ready: cmd_topic={self.cmd_topic}, '
            f'hz={self.hz}, move_speed_y={self.move_speed_y} m/s, '
            f'move_duration={self.move_duration} s'
        )

    def handle_open_crossing(self, request: SetBool.Request, response: SetBool.Response):
        """ 서비스 호출 시 True면 y축으로 move_duration 동안 이동 """
        if not request.data:
            response.success = True
            response.message = 'No-op (data was False).'
            return response

        now = self.get_clock().now()
        if self.active and self.active_until is not None and now < self.active_until:
            # 이미 동작 중이면 재진입 방지
            remaining = (self.active_until - now).nanoseconds / 1e9
            response.success = False
            response.message = f'Already moving. {remaining:.2f}s left.'
            return response

        self.active = True
        self.active_until = now + Duration(seconds=self.move_duration)
        response.success = True
        response.message = f'Move started for {self.move_duration:.2f}s at {self.move_speed_y:.2f} m/s (y).'
        self.get_logger().info(response.message)
        return response

    def on_timer(self):
        msg = Twist()

        if self.active and self.active_until is not None:
            now = self.get_clock().now()
            if now < self.active_until:
                # y축으로만 이동 (planar_move는 x/y/yaw만 사용)
                msg.linear.x = self.move_speed_y
            else:
                # 종료
                self.active = False
                self.active_until = None
                # msg는 기본 0 -> 정지
                self.get_logger().info('Move finished. Stopping.')

        # 필요시 기본 동작(예: 원운동)을 넣고 싶으면 여기에 else 분기를 추가
        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = GateBoxController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
