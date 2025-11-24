import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import math

class LidarArrayRotator(Node):
    def __init__(self):
        super().__init__('lidar_array_rotator')

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',  # 원본 토픽
            self.scan_callback,
            qos_profile)

        self.publisher = self.create_publisher(LaserScan, '/scan_rotated', 10)

    def scan_callback(self, msg):
        # 1. 데이터 개수 확인
        count = len(msg.ranges)
        
        # 2. 반 바퀴(180도)에 해당하는 인덱스 개수 계산
        # (전체 360도라고 가정하고 절반을 계산)
        mid_index = count // 2

        # 3. 배열 회전 (List Slicing) - 핵심!
        # [뒤쪽 데이터 ... 앞쪽 데이터] -> [앞쪽 데이터 ... 뒤쪽 데이터]
        # 뒤에 있던 절반(실제 앞)을 앞으로 가져오고, 앞에 있던 절반(실제 뒤)을 뒤로 보냄
        new_ranges = msg.ranges[mid_index:] + msg.ranges[:mid_index]
        
        # (필요시) 강도(intensities) 데이터도 똑같이 회전
        if msg.intensities:
            new_intensities = msg.intensities[mid_index:] + msg.intensities[:mid_index]
            msg.intensities = new_intensities

        # 4. 데이터 교체
        msg.ranges = new_ranges

        # 5. 프레임 ID 변경
        # 이제 데이터의 0번 인덱스가 '로봇 앞'이 되었으므로, base_link에 바로 붙입니다.
        msg.header.frame_id = 'base_link'

        # 6. 각도 범위 재설정 (중요!)
        # 데이터 순서를 바꿨으니, 이 데이터가 -180도 ~ +180도를 커버한다고 명시해줍니다.
        # 이렇게 해야 RViz에서 앞쪽(0도)을 기준으로 부채꼴이 이쁘게 펼쳐집니다.
        msg.angle_min = -math.pi
        msg.angle_max = math.pi
        
        # 발행
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = LidarArrayRotator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
