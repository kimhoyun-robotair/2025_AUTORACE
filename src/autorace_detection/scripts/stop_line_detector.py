#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv_bridge

import cv2
import numpy

class StopLineTracker:
    def __init__(self):
        self._delta = None

    def process(self, img: numpy.ndarray) -> None:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_white = numpy.array([0, 0, 200])
        upper_white = numpy.array([180, 255, 255])
        self._delta = None

        # 흰색 식별
        mask = cv2.inRange(hsv, lower_white, upper_white)

        h, w, d = img.shape
        search_top = int(13*h / 40)
        search_bot = int(25*h/40)

        # 마스킹
        mask[0:search_top, 0:w] = 0
        mask[search_bot:h, 0:w] = 0
        mask[0:h, 0:int(2 * w / 5)] = 0
        mask[0:h, int(3 * w / 5):w] = 0

        # 정지선 검출
        M = cv2.moments(mask)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(img, (cx, cy), 20, (0, 0, 255), -1)
            # BEGIN CONTROL
            err = abs(cx - w / 2)
            self._delta = err
            # END CONTROL

        # 카메라에서 오는 이미지 정보 띄워줌
        cv2.imshow("stop_window", img)
        cv2.imshow("stop_mask", mask)
        cv2.waitKey(3)

        # Decorator
        @property
        def _delta(self):
            return self._delta

class StopLineDetector(Node):
    def __init__(self, stop_line_tracker: StopLineTracker):
        super().__init__('stop_line_detector')

        # 지정된 차량 정보를 받아올 subscription
        self.stop_line_image_subscription_ = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.stop_line_image_callback,
            10
        )

        # 정지선 감지 정보를 전달하는 publisher
        self.stop_issue_publisher_ = self.create_publisher(
            String,
            'stop_issue',
            10
        )

        self.stop_line_image_subscription_ = None
        self.stop_line_tracker = stop_line_tracker
        self.bridge = cv_bridge.CvBridge()

    def stop_line_image_callback(self, image: Image):
        # ros image를 opencv image로 변환
        img = self.bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')

        # 이미지를 기반으로 정지선 검출
        self.stop_line_tracker.process(img)

        # 정지선 확인
        if self.stop_line_tracker._delta is not None and self.stop_line_tracker._delta < 3.0:
            msg = String()
            msg.data = '정지'
            self.stop_issue_publisher_.publish(msg)
        else:
            msg = String()
            msg.data = ''
            self.stop_issue_publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    stop_line_tracker = StopLineTracker()
    stop_line_detector = StopLineDetector(stop_line_tracker)

    rclpy.spin(stop_line_detector)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    stop_line_detector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()