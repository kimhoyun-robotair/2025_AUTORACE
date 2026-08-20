import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from ackermann_msgs.msg import AckermannDriveStamped
import math

class TwistToAckermann(Node):
    def __init__(self):
        super().__init__('twist_to_ackermann')
        
        # --- 설정 ---
        self.wheelbase = 0.33
        self.frame_id = "base_link"
        
        self.subscription = self.create_subscription(
            Twist,
            '/cmd_vel', 
            self.listener_callback,
            10)
            
        self.publisher = self.create_publisher(
            AckermannDriveStamped, 
            '/line_drive', 
            10)

    def listener_callback(self, msg):
        v = msg.linear.x
        w = msg.angular.z 

        # 1. 조향각 계산 (여기서는 원래 속도 v를 써야 핸들 방향이 맞습니다!)
        if abs(v) < 0.1: 
            steering = 0.0
        else:
            steering = math.atan((self.wheelbase * w) / v)

        # 메시지 생성
        ack_msg = AckermannDriveStamped()
        ack_msg.header.stamp = self.get_clock().now().to_msg()
        ack_msg.header.frame_id = self.frame_id
        
        # 2. 속도 방향 뒤집기 (하드웨어 모터 방향이 반대인 경우 보정)
        ack_msg.drive.speed = -v   # <--- 여기에 마이너스(-)를 붙여주세요!
        
        # 조향각은 그대로 넣습니다
        ack_msg.drive.steering_angle = steering
        
        self.publisher.publish(ack_msg)

def main(args=None):
    rclpy.init(args=args)
    node = TwistToAckermann()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()