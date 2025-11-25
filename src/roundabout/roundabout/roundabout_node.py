#!/usr/bin/env python3
import math
from enum import IntEnum, auto
from collections import deque
from typing import List, Tuple, Optional

import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu
from geometry_msgs.msg import Twist, Pose2D


# ===== tf_transformations 없이 직접 구현한 euler_from_quaternion =====
def euler_from_quaternion(quat):
    """
    quat: [x, y, z, w]
    return: (roll, pitch, yaw) [rad]
    """
    x, y, z, w = quat

    # roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    # pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = max(min(t2, 1.0), -1.0)  # clamp
    pitch_y = math.asin(t2)

    # yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z
# =======================================================================


class RoundaboutState(IntEnum):
    DETECT_PILLAR = auto()      # 2
    TRACK_OTHER_CARS = auto()   # 3
    WAIT_GAP = auto()           # 4
    ENTER = auto()              # 5
    CRUISE_AROUND = auto()      # 6
    EXIT_SEARCH = auto()        # 7
    EXIT = auto()               # 8
    RECOVERY = auto()           # 9 (옵션)


class RoundaboutNode(Node):
    def __init__(self):
        super().__init__('roundabout_node')

        # === 기본 파라미터 로드 ===
        self.declare_parameters('', [
            ('pillar_diameter_max', 0.58),
            ('pillar_tracking_window_sec', 1.0),
            ('roundabout_radius_lane_center', 1.5),
            ('roundabout_entry_speed', 0.4),
            ('roundabout_cruise_speed', 0.6),
            ('wheelbase', 0.26),
            ('track_width', 0.18),
            ('wheel_radius', 0.05),
            ('other_car_length', 0.35),
            ('other_car_width', 0.20),
            ('safe_front_distance', 0.7),
            ('safe_rear_distance', 0.5),
            ('stop_distance_hard', 0.4),
            ('k_lane_y', 1.2),
            ('k_lane_heading', 0.8),
            ('k_radius', 0.5),
            ('max_steer_angle', 0.5),
            ('max_accel', 1.0),
            ('exit_angle_deg', 180.0),
            ('exit_angle_tolerance_deg', 40.0),
            ('control_hz', 30.0),
            ('front_sector_deg', 60.0),
            ('car_frame_id', 'base_link'),
        ])

        p = self.get_parameter
        self.pillar_diameter_max = p('pillar_diameter_max').value
        self.pillar_tracking_window_sec = p('pillar_tracking_window_sec').value
        self.roundabout_radius_lane_center = p('roundabout_radius_lane_center').value
        self.roundabout_entry_speed = p('roundabout_entry_speed').value
        self.roundabout_cruise_speed = p('roundabout_cruise_speed').value
        self.wheelbase = p('wheelbase').value
        self.track_width = p('track_width').value
        self.wheel_radius = p('wheel_radius').value
        self.other_car_length = p('other_car_length').value
        self.other_car_width = p('other_car_width').value
        self.safe_front_distance = p('safe_front_distance').value
        self.safe_rear_distance = p('safe_rear_distance').value
        self.stop_distance_hard = p('stop_distance_hard').value
        self.k_lane_y = p('k_lane_y').value
        self.k_lane_heading = p('k_lane_heading').value
        self.k_radius = p('k_radius').value
        self.max_steer_angle = p('max_steer_angle').value
        self.max_accel = p('max_accel').value
        self.exit_angle_deg = p('exit_angle_deg').value
        self.exit_angle_tol_deg = p('exit_angle_tolerance_deg').value
        self.front_sector_deg = p('front_sector_deg').value
        self.control_hz = p('control_hz').value

        # --- pillar detection params ---
        self.declare_parameters('', [
            ('pillar_static_window_sec', 0.8),
            ('pillar_min_samples_per_beam', 5),
            ('pillar_max_std', 0.05),
            ('pillar_search_angle_deg', 120.0),
            ('pillar_min_radius', 0.2),
            ('pillar_max_radius', 5.0),
        ])

        self.pillar_static_window_sec = self.get_parameter('pillar_static_window_sec').value
        self.pillar_min_samples_per_beam = int(self.get_parameter('pillar_min_samples_per_beam').value)
        self.pillar_max_std = float(self.get_parameter('pillar_max_std').value)
        self.pillar_search_angle_deg = float(self.get_parameter('pillar_search_angle_deg').value)
        self.pillar_min_radius = float(self.get_parameter('pillar_min_radius').value)
        self.pillar_max_radius = float(self.get_parameter('pillar_max_radius').value)

        # LiDAR 스캔 버퍼 시간창
        self.scan_window_sec = self.pillar_static_window_sec

        # === ROS I/O ===
        self.sub_scan = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.sub_imu = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)
        self.sub_lane = self.create_subscription(
            Pose2D, '/lane_pose', self.lane_pose_callback, 10)

        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)

        # === 상태 변수 ===
        self.state = RoundaboutState.DETECT_PILLAR

        # LiDAR 데이터 버퍼 (기둥/차량 추적용)
        self.scan_buffer: deque[Tuple[LaserScan, float]] = deque()  # (msg, stamp_sec)

        # 기둥 정보
        self.pillar_found = False
        self.pillar_center_in_lidar: Optional[Tuple[float, float]] = None  # (x, y)
        self.roundabout_radius_target = self.roundabout_radius_lane_center

        # 차량(다른 차) 정보 (아주 단순한 표현: (r, theta) 목록)
        self.other_cars_polar: List[Tuple[float, float]] = []

        # IMU yaw
        self.yaw = 0.0
        self.yaw_enter_ref = None    # 회전 교차로 진입 시 yaw 기준값 (6시)
        self.yaw_deg = 0.0

        # lane pose (BEV 결과)
        self.lane_y = 0.0            # lateral offset (+왼쪽, -오른쪽)
        self.lane_heading_err = 0.0  # 도로 기준 heading error

        # 시간 관리
        self.last_time = self.get_clock().now()

        # 메인 제어 루프
        period = 1.0 / self.control_hz
        self.control_timer = self.create_timer(period, self.control_loop)

        self.get_logger().info('RoundaboutNode initialized. Starting in DETECT_PILLAR state.')

    # =========================
    # 콜백들
    # =========================
    def scan_callback(self, msg: LaserScan):
        now = self.get_clock().now()
        now_sec = now.seconds_nanoseconds()[0] + now.seconds_nanoseconds()[1] * 1e-9
        self.scan_buffer.append((msg, now_sec))

        # 오래된 스캔 제거
        while self.scan_buffer and (now_sec - self.scan_buffer[0][1] > self.scan_window_sec):
            self.scan_buffer.popleft()

    def imu_callback(self, msg: Imu):
        q = msg.orientation
        quat = [q.x, q.y, q.z, q.w]
        roll, pitch, yaw = euler_from_quaternion(quat)
        self.yaw = yaw
        self.yaw_deg = math.degrees(yaw)

    def lane_pose_callback(self, msg: Pose2D):
        self.lane_y = msg.x
        self.lane_heading_err = msg.theta

    # =========================
    # 메인 제어 루프
    # =========================
    def control_loop(self):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now
        if dt <= 0.0:
            dt = 1.0 / self.control_hz

        # 상태 머신 업데이트
        if self.state == RoundaboutState.DETECT_PILLAR:
            self.update_detect_pillar(dt)
        elif self.state == RoundaboutState.TRACK_OTHER_CARS:
            self.update_track_other_cars(dt)
        elif self.state == RoundaboutState.WAIT_GAP:
            self.update_wait_gap(dt)
        elif self.state == RoundaboutState.ENTER:
            self.update_enter(dt)
        elif self.state == RoundaboutState.CRUISE_AROUND:
            self.update_cruise_around(dt)
        elif self.state == RoundaboutState.EXIT_SEARCH:
            self.update_exit_search(dt)
        elif self.state == RoundaboutState.EXIT:
            self.update_exit(dt)
        elif self.state == RoundaboutState.RECOVERY:
            self.update_recovery(dt)
        else:
            self.get_logger().warn(f'Unknown state: {self.state}')
            self.publish_stop()

    # =========================
    # 상태별 업데이트 함수들
    # =========================
    def update_detect_pillar(self, dt: float):
        """2. 중앙 기둥 인식"""
        if not self.scan_buffer:
            self.publish_stop()
            return

        pillar = self.estimate_pillar_from_scans()

        if pillar is not None:
            self.pillar_center_in_lidar = pillar  # (x, y)
            self.pillar_found = True
            self.get_logger().info(
                f'Pillar detected at (x={pillar[0]:.2f}, y={pillar[1]:.2f}) in lidar frame.'
            )
            self.state = RoundaboutState.TRACK_OTHER_CARS
        else:
            # 기둥 못 찾으면 정지 유지
            self.publish_stop()

    def update_track_other_cars(self, dt: float):
        """3. 회전하는 차량 궤적 파악 (현재는 매우 단순 버전)"""
        if not self.pillar_found:
            self.state = RoundaboutState.DETECT_PILLAR
            return

        self.update_other_cars_simple()

        # 지금 구현에서는 "차를 관찰했다" 정도만 보고 그냥 WAIT_GAP으로 넘어감
        self.state = RoundaboutState.WAIT_GAP
        self.get_logger().info('Switched to WAIT_GAP.')

    def update_wait_gap(self, dt: float):
        """4. 안전한 gap 생길 때까지 대기 (간단 버전: 전방 sector에 차가 없으면 진입)"""
        self.update_other_cars_simple()

        front_clear = self.is_front_sector_clear()

        if front_clear:
            # 진입 준비
            self.yaw_enter_ref = self.yaw  # 6시 방향 yaw 기준 저장
            self.get_logger().info('Front sector clear. ENTERing roundabout.')
            self.state = RoundaboutState.ENTER
        else:
            self.publish_stop()

    def update_enter(self, dt: float):
        """5. 회전 교차로 진입"""
        v_cmd = self.roundabout_entry_speed

        # lane + pillar 기반 조향
        steer = self.compute_steering_command(use_radius=True)

        # 앞차와 거리 체크
        self.update_other_cars_simple()
        if not self.is_front_sector_clear():
            v_cmd = 0.0

        self.publish_cmd(v_cmd, steer)

        # 일정 yaw 변화 → roundabout 내부라고 판단
        if self.yaw_enter_ref is not None:
            delta_deg = self.angle_diff_deg(
                math.degrees(self.yaw),
                math.degrees(self.yaw_enter_ref)
            )
            if abs(delta_deg) > 30.0:  # 예: 30도 이상 돌면 내부
                self.get_logger().info('Entered roundabout. Switching to CRUISE_AROUND.')
                self.state = RoundaboutState.CRUISE_AROUND

    def update_cruise_around(self, dt: float):
        """6. 라운드어바웃 내부에서 회전 주행"""
        v_cmd = self.roundabout_cruise_speed

        steer = self.compute_steering_command(use_radius=True)

        # 앞차와 거리
        self.update_other_cars_simple()
        front_clear = self.is_front_sector_clear()

        if not front_clear:
            v_cmd = 0.0

        self.publish_cmd(v_cmd, steer)

        # yaw 각도 기반으로 EXIT_SEARCH 진입 시점 판단
        if self.yaw_enter_ref is not None:
            delta_deg = self.angle_diff_deg(
                math.degrees(self.yaw),
                math.degrees(self.yaw_enter_ref)
            )
            if self.is_in_exit_region(delta_deg):
                self.get_logger().info(
                    f'In exit region (Δψ≈{delta_deg:.1f}deg). Switching to EXIT_SEARCH.'
                )
                self.state = RoundaboutState.EXIT_SEARCH

    def update_exit_search(self, dt: float):
        """7. 출구 차선 패턴 탐색 (현재는 yaw 각도만 사용한 매우 단순 버전)"""
        if self.yaw_enter_ref is None:
            self.state = RoundaboutState.CRUISE_AROUND
            return

        delta_deg = self.angle_diff_deg(
            math.degrees(self.yaw),
            math.degrees(self.yaw_enter_ref)
        )

        # 단순 버전: exit region 안에 있으면 바로 EXIT 모드
        if self.is_in_exit_region(delta_deg):
            self.get_logger().info('EXIT lane assumed. Switching to EXIT.')
            self.state = RoundaboutState.EXIT
        else:
            self.state = RoundaboutState.CRUISE_AROUND

        v_cmd = self.roundabout_cruise_speed
        steer = self.compute_steering_command(use_radius=False)  # 출구 찾을 땐 pillar 비중 줄이기
        self.publish_cmd(v_cmd, steer)

    def update_exit(self, dt: float):
        """8. 출구 차선을 따라 회전 교차로에서 빠져나가기"""
        v_cmd = self.roundabout_cruise_speed
        steer = self.compute_steering_command(use_radius=False)

        self.update_other_cars_simple()
        if not self.is_front_sector_clear():
            v_cmd = 0.0

        self.publish_cmd(v_cmd, steer)
        # TODO: 일정 조건(라운드어바웃 완전히 벗어남)에서 다음 미션 상태로 전환하는 로직 추가 가능

    def update_recovery(self, dt: float):
        """9. 막혔을 때 다시 WAIT_GAP으로 돌아가는 등 회복 로직 (현재는 정지만)"""
        self.publish_stop()
        # TODO: timeout / 조건 기반 WAIT_GAP 복귀 구현 가능

    # =========================
    # 보조 함수들
    # =========================
    def estimate_pillar_from_scans(self) -> Optional[Tuple[float, float]]:
        """
        scan_buffer를 이용해 정적인 작은 클러스터(기둥)를 찾는다.
        """
        if not self.scan_buffer:
            return None

        scans = [s for (s, t) in self.scan_buffer]
        num_scans = len(scans)
        if num_scans < self.pillar_min_samples_per_beam:
            return None

        latest_scan = scans[-1]
        n_beams = len(latest_scan.ranges)

        ranges_per_beam = [[] for _ in range(n_beams)]

        # 각 beam index별로 거리값 모으기
        for scan in scans:
            for i, r in enumerate(scan.ranges):
                if scan.range_min < r < scan.range_max:
                    ranges_per_beam[i].append(r)

        angle_min = latest_scan.angle_min
        angle_inc = latest_scan.angle_increment
        half_search_rad = math.radians(self.pillar_search_angle_deg) / 2.0

        static_mask = [False] * n_beams
        mean_range = [0.0] * n_beams

        for i in range(n_beams):
            rs = ranges_per_beam[i]
            if len(rs) < self.pillar_min_samples_per_beam:
                continue

            r_arr = np.array(rs, dtype=np.float32)
            mu = float(r_arr.mean())
            sigma = float(r_arr.std())

            angle_i = angle_min + i * angle_inc

            if (
                self.pillar_min_radius < mu < self.pillar_max_radius and
                abs(angle_i) <= half_search_rad and
                sigma <= self.pillar_max_std
            ):
                static_mask[i] = True
                mean_range[i] = mu

        # 정적 beam들을 연속 구간으로 묶어 클러스터 생성
        clusters = []  # list of (start_idx, end_idx)
        in_cluster = False
        cluster_start = 0

        for i in range(n_beams):
            if static_mask[i] and not in_cluster:
                in_cluster = True
                cluster_start = i
            elif not static_mask[i] and in_cluster:
                in_cluster = False
                clusters.append((cluster_start, i - 1))
        if in_cluster:
            clusters.append((cluster_start, n_beams - 1))

        if not clusters:
            return None

        best_cluster = None
        best_cluster_r = float('inf')

        for (i_start, i_end) in clusters:
            if i_end <= i_start:
                continue

            r_vals = []
            theta_vals = []

            for i in range(i_start, i_end + 1):
                if static_mask[i]:
                    r_vals.append(mean_range[i])
                    theta_vals.append(angle_min + i * angle_inc)

            if not r_vals:
                continue

            r_mean = float(np.mean(r_vals))
            theta_mean = float(np.mean(theta_vals))

            theta_min = min(theta_vals)
            theta_max = max(theta_vals)
            dtheta = theta_max - theta_min

            # 폭 근사
            width_approx = r_mean * abs(dtheta)

            if width_approx <= self.pillar_diameter_max * 1.5:
                if r_mean < best_cluster_r:
                    best_cluster_r = r_mean
                    best_cluster = (r_mean, theta_mean)

        if best_cluster is None:
            return None

        r_pillar, theta_pillar = best_cluster
        x = r_pillar * math.cos(theta_pillar)
        y = r_pillar * math.sin(theta_pillar)

        return (x, y)

    def update_other_cars_simple(self):
        """
        다른 차량을 아주 단순하게 표현:
          - front_sector 안에 있는 장애물을 차량 후보로 본다.
        """
        self.other_cars_polar.clear()
        if not self.scan_buffer:
            return

        scan, t = self.scan_buffer[-1]
        half_sector_rad = math.radians(self.front_sector_deg) / 2.0
        angle = scan.angle_min

        for r in scan.ranges:
            if scan.range_min < r < scan.range_max:
                if -half_sector_rad <= angle <= half_sector_rad:
                    self.other_cars_polar.append((r, angle))
            angle += scan.angle_increment

    def is_front_sector_clear(self) -> bool:
        """앞쪽 sector에 stop_distance_hard보다 가까운 장애물이 있는지 검사"""
        for r, ang in self.other_cars_polar:
            if r < self.stop_distance_hard:
                return False
        return True

    def compute_steering_command(self, use_radius: bool = True) -> float:
        """
        lane_y, lane_heading_err, pillar 거리(반경)를 사용해서 조향 명령 생성.
        use_radius=True일 때는 기둥 반경 오차도 포함.
        """
        steer = 0.0

        # 차선 기반
        steer += self.k_lane_y * self.lane_y
        steer += self.k_lane_heading * self.lane_heading_err

        # 기둥 반경 기반
        if use_radius and self.pillar_center_in_lidar is not None:
            x, y = self.pillar_center_in_lidar
            d_center = math.hypot(x, y)
            error_r = d_center - self.roundabout_radius_target
            steer += self.k_radius * error_r

        steer = max(-self.max_steer_angle, min(self.max_steer_angle, steer))
        return steer

    def is_in_exit_region(self, delta_deg: float) -> bool:
        """6시→12시 출구 근처 각도 범위 안에 있는지 확인"""
        center = self.exit_angle_deg   # 180deg
        tol = self.exit_angle_tol_deg  # 예: 40deg → 140~220deg
        return (center - tol) <= delta_deg <= (center + tol)

    @staticmethod
    def angle_diff_deg(a: float, b: float) -> float:
        """두 각도(도)를 -180~180 범위 차이로 변환"""
        d = a - b
        while d > 180.0:
            d -= 360.0
        while d < -180.0:
            d += 360.0
        return d

    def publish_cmd(self, v: float, steer: float):
        msg = Twist()
        msg.linear.x = v
        msg.angular.z = steer
        self.pub_cmd.publish(msg)

    def publish_stop(self):
        self.publish_cmd(0.0, 0.0)


def main(args=None):
    rclpy.init(args=args)
    node = RoundaboutNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

