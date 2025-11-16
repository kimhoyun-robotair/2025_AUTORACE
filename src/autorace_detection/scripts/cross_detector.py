#!/usr/bin/env python3
"""
RectStopGoNode (ROS2 Humble, Python)

목적:
- LaserScan(/scan)으로부터 전방(+x) 장애물 "전면(edge) 거리"를 추정하고,
  임계거리(d_stop) 이하이면 False(정지), 초과면 True(주행가능)를 /obstacle/clear에 Bool로 퍼블리시.
- 디버깅을 위해 RViz MarkerArray로 클러스터 점, 피팅된 사각형, 전면 에지, 텍스트를 시각화.

핵심 처리 파이프라인:
1) LaserScan -> 극좌표(r, θ) -> 직교좌표(x, y) (base_link 기준 가정)
2) O(N) Range-Jump 클러스터링(인접 빔 간 거리 점프 기반 분할)
3) 가장 가까운 클러스터 선택 (최근접 x 최소값)
4) OpenCV의 minAreaRect로 최소면적 사각형 피팅 → 사각형 4꼭지점(boxPoints)
5) 전면(edge)의 x 최소값을 전면거리 d_front로 사용(중심점보다 충돌 판정에 보수적)
6) 선택적 안정화(EMA, 히스테리시스, M/N 투표)
7) Bool publish + MarkerArray publish

사용 라이브러리/ROS 메시지:
- numpy: 벡터/행렬 수치 계산, 정렬, norm 등
- math: 삼각함수/기본 수학
- OpenCV(cv2): minAreaRect(사각형 피팅), boxPoints(사각형 꼭지점 추출)
- rclpy: ROS2 Python 클라이언트 라이브러리(노드, 파라미터, QoS, pub/sub)
- sensor_msgs/LaserScan: LiDAR 스캔 입력
- std_msgs/Bool: 주행가능 여부(Boolean)
- visualization_msgs/Marker, MarkerArray: RViz 시각화
- geometry_msgs/Point: 마커 좌표 표현
"""

import math
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


# --------------------------- 유틸리티 함수 ---------------------------

def polar_to_xy(r: float, a: float):
    """
    극좌표(r, a) -> 직교좌표(x, y) 변환.

    매개변수
    - r: range(거리). LaserScan.ranges의 요소.
    - a: angle(라디안). LaserScan.angle_min + i*angle_increment.

    반환
    - (x, y): base_link를 원점으로 +x 전방, +y 좌측(ROS 기본 회전방향 기준) 좌표.

    구현 상세
    - math.cos/sin 사용(표준 라이브러리). numpy 대신 math를 써서 단건 계산을 가볍게.
    """
    return r * math.cos(a), r * math.sin(a)


# --------------------------- 메인 노드 ---------------------------

class RectStopGoNode(Node):
    """
    LaserScan을 받아서 전방 사각형 피팅 기반 전면거리(d_front)를 계산하고,
    임계값(d_stop) 기준으로 Bool을 퍼블리시하는 ROS2 노드.

    주요 멤버
    - 파라미터: scan_topic, out_topic, marker_topic, d_stop, alpha/beta(클러스터링), min_pts/max_diameter,
                EMA/히스테리시스/MN 투표 옵션, 사각형 폭/길이 타당성 범위
    - 퍼블리셔: pub_bool(std_msgs/Bool), pub_mark(visualization_msgs/MarkerArray)
    - 내부 상태: d_smooth(EMA 상태), last_bool(직전 판정), vote_true/vote_false(M/N 투표 카운터)
    """
    def __init__(self):
        super().__init__('rect_stop_go_node')

        # ------------------- 파라미터 선언 -------------------
        # 토픽/입출력
        self.declare_parameter('scan_topic', '/scan')               # 입력 LiDAR 스캔 토픽(sensor_msgs/LaserScan)
        self.declare_parameter('out_topic', '/obstacle/clear')      # 출력 Bool 토픽(True=주행가능, False=정지)
        self.declare_parameter('marker_topic', '/rect_stop_go/markers')  # RViz MarkerArray 토픽

        # 임계값/판단
        self.declare_parameter('d_stop', 1.2)                       # [m] 전면(edge) 추정거리 임계

        # 클러스터링(Range-Jump 방식)
        self.declare_parameter('alpha', 0.05)                       # [m] 근거리 기준 점프 임계 기본항(민감도)
        self.declare_parameter('beta',  0.03)                       # [m/m] 거리 의존 보정(멀수록 점 간격 커지는 현상 보정)
        self.declare_parameter('min_pts', 8)                        # [개] 클러스터 최소 포인트 수(노이즈 억제)
        self.declare_parameter('max_diameter', 1.5)                 # [m] 클러스터 최대 직경(벽 혼입 제거)

        # 신호 안정화 옵션
        self.declare_parameter('use_ema', False)                    # EMA 평활 사용 여부(거리 잡음 완화)
        self.declare_parameter('ema_lambda', 0.7)                   # EMA 계수 λ (0~1, 클수록 느리게/안정)
        self.declare_parameter('use_hysteresis', False)             # 히스테리시스 사용 여부(경계 깜빡임 방지)
        self.declare_parameter('hyst_margin', 0.4)                  # [m] 히스테리시스 마진(d_go = d_stop + margin)
        self.declare_parameter('use_mn_vote', False)                # M/N 투표(다수결) 사용 여부
        self.declare_parameter('M', 2)                              # [프레임] 투표 임계 M (N 중 M 이상 시 확정)
        self.declare_parameter('N', 3)                              # [프레임] 투표 윈도 길이 N

        # 사각형 타당성(폭/길이 범위)
        self.declare_parameter('width_min', 0.3)                    # [m] 짧은 변 최소
        self.declare_parameter('width_max', 2.5)                    # [m] 짧은 변 최대
        self.declare_parameter('length_min', 0.3)                   # [m] 긴 변 최소
        self.declare_parameter('length_max', 3.5)                   # [m] 긴 변 최대

        # 라이다 FOV 제한
        self.declare_parameter('fov_min_deg', -10.0)   # [deg] 하한각
        self.declare_parameter('fov_max_deg',  10.0)   # [deg] 상한각

        # ------------------- 파라미터 해석/저장 -------------------
        gp = lambda n: self.get_parameter(n).value
        self.scan_topic   = gp('scan_topic')
        self.out_topic    = gp('out_topic')
        self.marker_topic = gp('marker_topic')

        self.d_stop      = float(gp('d_stop'))
        self.alpha       = float(gp('alpha'))
        self.beta        = float(gp('beta'))
        self.min_pts     = int(gp('min_pts'))
        self.max_diam    = float(gp('max_diameter'))

        self.use_ema     = bool(gp('use_ema'))
        self.ema_lambda  = float(gp('ema_lambda'))
        self.use_hyst    = bool(gp('use_hysteresis'))
        self.hyst_margin = float(gp('hyst_margin'))
        self.use_mn      = bool(gp('use_mn_vote'))
        self.M           = int(gp('M'))
        self.N           = int(gp('N'))

        self.width_min   = float(gp('width_min'))
        self.width_max   = float(gp('width_max'))
        self.length_min  = float(gp('length_min'))
        self.length_max  = float(gp('length_max'))

        self.fov_min = math.radians(float(gp('fov_min_deg')))
        self.fov_max = math.radians(float(gp('fov_max_deg')))

        # ------------------- QoS/IO 설정 -------------------
        # BEST_EFFORT: LaserScan 생산자가 best effort일 때 QoS 불일치 방지
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        # 구독자: /scan -> scan_cb로 콜백
        self.sub = self.create_subscription(LaserScan, self.scan_topic, self.scan_cb, qos)
        # 퍼블리셔: 결론(Boolean)과 RViz MarkerArray
        self.pub_bool = self.create_publisher(Bool, self.out_topic, 10)
        self.pub_mark = self.create_publisher(MarkerArray, self.marker_topic, 10)

        # ------------------- 내부 상태 -------------------
        self.d_smooth = None     # EMA 누적 상태(없으면 None)
        self.last_bool = None    # 직전 결정(True/False)
        self.vote_true = 0       # M/N 투표용 True 카운터
        self.vote_false = 0      # M/N 투표용 False 카운터

        self.marker_ns = "rect_stop_go"  # RViz 마커 namespace
        self.get_logger().info(
            f"RectStopGoNode: scan={self.scan_topic}, out={self.out_topic}, marker={self.marker_topic}, "
            f"d_stop={self.d_stop:.2f}, alpha={self.alpha:.3f}, beta={self.beta:.3f}, min_pts={self.min_pts}"
        )

    # -------------------- Core pipeline --------------------
    def scan_cb(self, msg: LaserScan):
        """
        LaserScan 콜백(ROS2 구독). 전체 파이프라인을 오케스트레이션.

        단계:
        1) scan_to_xy_array: LaserScan(ranges, angle_min/max, angle_increment) -> (x,y) 배열
        2) range_jump_cluster: 인접 빔 간 거리 점프 임계로 점군 분할(O(N))
        3) pick_nearest_cluster: +x(전방) 방향으로 가장 가까운 클러스터 선택
        4) front_edge_distance_minrect: OpenCV minAreaRect로 사각형 피팅 -> boxPoints -> x 최소값을 전면거리
           - 사각형 타당성(폭/길이 범위) 불만족 시 포인트 최근접 x로 폴백
        5) 옵션: EMA(평활), 히스테리시스(경계 깜빡임 완화), M/N 투표(다수결 안정화)
        6) Bool Publish + RViz Marker Publish
        """
        # LaserScan.frame_id가 비어있으면 기본 "base_link" 사용(시각화 프레임)
        frame_id = msg.header.frame_id or "base_link"

        # 1) LaserScan -> XY 점군
        pts = self.scan_to_xy_array(msg)
        if pts.shape[0] == 0:
            # 포인트가 없으면 '멀다(True)'로 간주하고 마커 삭제
            self.publish_bool(True)
            self.publish_clear_markers(frame_id, msg.header.stamp)
            return

        # 2) Range-Jump 클러스터링
        clusters = self.range_jump_cluster(pts, self.alpha, self.beta)
        if not clusters:
            # 클러스터가 전혀 형성되지 않으면 멀다(True)
            self.publish_bool(True)
            self.publish_clear_markers(frame_id, msg.header.stamp)
            return

        # 3) 가장 가까운 클러스터 선택
        target = self.pick_nearest_cluster(clusters)
        if target is None or target.shape[0] < self.min_pts:
            # 최근접 클러스터가 없거나 점이 너무 적으면 멀다(True)
            self.publish_bool(True)
            self.publish_clear_markers(frame_id, msg.header.stamp)
            return

        # 4) 사각형 피팅 기반 전면거리(d_front) 추정
        d_front, box = self.front_edge_distance_minrect(target)

        # 수치 실패/NaN 방지: 폴백(점군 최근접 x)
        if not np.isfinite(d_front):
            d_front = float(np.min(target[:, 0]))
            box = None

        # 5) 신호 안정화 단계(옵션)
        # 5-1) EMA: 지수이동평균으로 d_front 평활
        d_eff = d_front
        if self.use_ema:
            self.d_smooth = d_front if self.d_smooth is None else \
                            self.ema_lambda * self.d_smooth + (1.0 - self.ema_lambda) * d_front
            d_eff = self.d_smooth

        # 5-2) 즉시판정: d_eff > d_stop 이면 True(주행가능), 아니면 False(정지)
        decision = d_eff > self.d_stop

        # 5-3) 히스테리시스: 이전 상태를 고려해 경계 근처 깜빡임 억제
        if self.use_hyst:
            # 직전 False(정지)였다면 d_stop + margin을 넘어야 True로 전환
            if self.last_bool is False and d_eff <= (self.d_stop + self.hyst_margin):
                decision = False
            # 직전 True(주행가능)라면 d_stop 초과 유지 시 True 지속
            elif self.last_bool is True and d_eff > self.d_stop:
                decision = True

        # 5-4) M/N 투표: 최근 N프레임 중 M회 이상 일치 시 확정(다수결)
        if self.use_mn:
            if decision:
                self.vote_true = min(self.vote_true + 1, self.N); self.vote_false = 0
            else:
                self.vote_false = min(self.vote_false + 1, self.N); self.vote_true = 0
            if self.vote_true >= self.M:
                decision = True
            elif self.vote_false >= self.M:
                decision = False
            else:
                # 아직 확정 임계 미달이면 이전 상태 유지
                decision = self.last_bool if self.last_bool is not None else decision

        # 6) 결과 퍼블리시 + 마커 퍼블리시
        self.publish_bool(decision)
        self.publish_markers(frame_id, msg.header.stamp, target, box, d_front, decision)

    # -------------------- Helpers: 변환/클러스터링/피팅 --------------------
    def _angle_in_fov(self, a: float) -> bool:
        """
        a(라디안)가 [self.fov_min, self.fov_max] 범위에 들어가는지 체크.
        - 모든 각도를 [-pi, pi]로 정규화한 뒤 비교
        - fov_min > fov_max 인 경우(경계가 ±pi를 넘나드는 래핑)도 지원
        """
        pi2 = 2.0 * math.pi

        def norm(x):  # [-pi, pi]로 정규화
            return (x + math.pi) % pi2 - math.pi

        a    = norm(a)
        amin = norm(self.fov_min)
        amax = norm(self.fov_max)

        if amin <= amax:
            return amin <= a <= amax
        else:
            # 래핑 구간: 예) amin=170°, amax=-170° 처럼 경계가 π를 넘어감
            return (a >= amin) or (a <= amax)

    def scan_to_xy_array(self, msg: LaserScan) -> np.ndarray:
        """
        LaserScan -> XY 점군 변환.

        구현 상세:
        - LaserScan.angle_min부터 angle_increment씩 증가시키며 ranges[i]를 (x,y)로 변환
        - np.isfinite로 유효 거리만 사용
        - r_max_allowed를 range_max로 클램프(일부 드라이버/시뮬은 매우 큰 값이 들어갈 수 있어 보호)
        - 반환: shape (N,2) float32 numpy 배열; N=유효 포인트 개수

        사용 API:
        - math.cos/sin
        - numpy array 구성
        """
        ang = msg.angle_min
        pts = []
        idx = 0
        r_max_allowed = min(getattr(msg, 'range_max', float('inf')), 1e6)  # 비정상 큰 값 방지
        while idx < len(msg.ranges):
            r = msg.ranges[idx]
            if np.isfinite(r) and (r > 0.05) and (r < r_max_allowed):
                if not self._angle_in_fov(ang):
                    idx += 1
                    ang += msg.angle_increment
                    continue
                x, y = polar_to_xy(r, ang)
                pts.append((x, y))
            idx += 1
            ang += msg.angle_increment
        return np.array(pts, dtype=np.float32) if pts else np.empty((0, 2), dtype=np.float32)

    def range_jump_cluster(self, pts: np.ndarray, alpha: float, beta: float):
        """
        O(N) Range-Jump 기반의 스캔라인 클러스터링.

        아이디어:
        - 각 포인트를 각도 기준으로 정렬(스캔 순서와 동일)
        - 인접 점 p, q 간의 유클리드 거리 ||q - p||가
          임계값(thresh = alpha + beta * max(||p||, ||q||))를 초과하면 "클러스터 경계"로 판단하여 분할
          * alpha: 근거리 기준 민감도
          * beta: 원거리에서 빔 간 공간 간격 증가를 보정

        추가 필터:
        - cluster_plausible로 최소 포인트 수(min_pts)와 최대 직경(max_diameter) 검사

        사용 API:
        - numpy.arctan2: 각도 계산
        - numpy.argsort: 각도 정렬
        - numpy.linalg.norm: 유클리드 거리
        """
        if pts.shape[0] < 2:
            return []
        ang = np.arctan2(pts[:, 1], pts[:, 0])
        order = np.argsort(ang)
        P = pts[order]

        clusters = []
        current = [P[0]]
        for i in range(P.shape[0] - 1):
            p, q = P[i], P[i + 1]
            r1 = np.linalg.norm(p)
            r2 = np.linalg.norm(q)
            thresh = alpha + beta * max(r1, r2)
            # 인접 점 간 거리 점프가 임계보다 크면 새 클러스터 시작
            if np.linalg.norm(q - p) > thresh:
                C = np.array(current)
                if self.cluster_plausible(C):
                    clusters.append(C)
                current = [q]
            else:
                current.append(q)
        # 마지막 클러스터 처리
        C = np.array(current)
        if self.cluster_plausible(C):
            clusters.append(C)
        return clusters

    def cluster_plausible(self, C: np.ndarray) -> bool:
        """
        클러스터 타당성 검사.

        - 포인트 수가 min_pts 미만이면 노이즈로 간주 → False
        - '최가까운 점'을 기준점으로 한 대략적 직경(최대거리)을 구해 max_diameter 초과 시 False
          (벽 일부가 섞여 클러스터가 과도하게 커지는 경우 방지)
        """
        if C.shape[0] < self.min_pts:
            return False
        # 가장 가까운 점(전방축 기준 x 최소)과의 최대 거리 -> 직경 근사
        diam = np.max(np.linalg.norm(C - C[np.argmin(C[:, 0])], axis=1))
        return diam <= self.max_diam

    def pick_nearest_cluster(self, clusters):
        """
        최근접 클러스터 선택.

        - 각 클러스터의 '가장 작은 x'(= 전방(+x) 축으로 가장 가까운 점)를 대표 거리로 사용
        - 가장 작은 값을 가지는 클러스터를 타깃으로 선택
        """
        best, best_d = None, float('inf')
        for C in clusters:
            d = float(np.min(C[:, 0]))
            if d < best_d:
                best, best_d = C, d
        return best

    def front_edge_distance_minrect(self, pts: np.ndarray):
        """
        사각형 피팅으로 전면(edge) 거리 계산.

        절차:
        1) cv2.minAreaRect(points) -> (center(x,y), (w,h), angle)
           - 점군을 가장 작은 면적의 회전 사각형으로 근사(기하학적 외접 사각형)
        2) cv2.boxPoints(rect) -> 4개 꼭지점 좌표(회전된 좌표계 고려)
        3) 4꼭지점의 x좌표들 중 최솟값(min(box[:,0]))을 전면(edge)의 거리로 사용
           - 중심점보다 보수적이며 실제 충돌/안전 판단에 유리
        4) 사각형의 짧은 변/긴 변이 사전에 정의한 타당성 범위에 들어오는지 검사
           - 범위를 벗어나면(= 벽 섞임/이상치) 최근접 포인트 기반 거리로 폴백

        사용 API:
        - cv2.minAreaRect, cv2.boxPoints (OpenCV)
        - numpy 연산
        """
        try:
            rect = cv2.minAreaRect(pts.astype(np.float32))  # ((cx,cy),(w,h),theta)
            box = cv2.boxPoints(rect)  # (4,2) 꼭지점들
            width, length = self._rect_sides(box)
            # 폭/길이 타당성 체크(너무 작거나 크면 박스로 보기 어려움)
            if not (self.width_min <= width <= self.width_max and
                    self.length_min <= length <= self.length_max):
                return float(np.min(pts[:, 0])), None
            d_front = float(np.min(box[:, 0]))  # 전면 edge ~ x 최소값
            return max(0.0, d_front), box
        except Exception as e:
            # 수치문제/특이치 등으로 피팅 실패 시 포인트 최근접 x로 폴백
            self.get_logger().warn(f"minAreaRect failed: {e}")
            return float(np.min(pts[:, 0])), None

    @staticmethod
    def _rect_sides(box: np.ndarray):
        """
        사각형 4꼭지점으로부터 서로 마주보는 변의 길이를 평균해
        '짧은 변'(width)과 '긴 변'(length)을 구한다.

        구현:
        - 변 길이: ||p_{i+1} - p_i|| (i=0..3, 순환)
        - 서로 마주보는 변(0-1 vs 2-3, 1-2 vs 3-0)의 평균을 각각 구한 뒤, 작은쪽/큰쪽을 (width, length)로 리턴.
        """
        e = [
            np.linalg.norm(box[1] - box[0]),
            np.linalg.norm(box[2] - box[1]),
            np.linalg.norm(box[3] - box[2]),
            np.linalg.norm(box[0] - box[3]),
        ]
        a, b = sorted([np.mean([e[0], e[2]]), np.mean([e[1], e[3]])])
        return a, b

    # -------------------- Publish: Bool & Markers --------------------

    def publish_bool(self, decision: bool):
        """
        최종 불리언 결과 퍼블리시 및 내부 상태 업데이트.

        - True  : clear(멀다) -> 주행가능
        - False : close(가깝다) -> 정지
        """
        msg = Bool(); msg.data = decision
        self.pub_bool.publish(msg)
        self.last_bool = decision

    def publish_clear_markers(self, frame_id: str, stamp):
        """
        RViz 마커 전부 삭제(화면 클리어).

        - Marker.DELETEALL 액션 사용.
        - RViz에 남아있는 이전 프레임 시각화를 지우고 싶을 때 호출.
        """
        arr = MarkerArray()
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = stamp
        m.ns = self.marker_ns
        m.id = 0
        m.action = Marker.DELETEALL
        arr.markers.append(m)
        self.pub_mark.publish(arr)

    def publish_markers(self, frame_id: str, stamp, pts: np.ndarray, box: np.ndarray, d_front: float, clear: bool):
        """
        RViz 디버깅용 마커 게시.

        구성:
        0) 클러스터 점(POINTS)
        1) 사각형 윤곽선(LINE_STRIP) - 피팅 성공 시에만
        2) 전면 에지(LINE_LIST) - x = min(box[:,0])
        3) 사각형 중심(SPHERE)
        4) 텍스트(TEXT_VIEW_FACING) - d_front 및 clear(True/False) 표시

        색상:
        - 클러스터: 하늘색
        - 사각형: 주황/빨강
        - 전면 에지: 노란색
        - 텍스트: clear면 녹색, 아니면 빨강
        """
        arr = MarkerArray()

        # (0) 클러스터 점
        arr.markers.append(self.mk_points(frame_id, stamp, 1, pts, rgba(0.2, 0.8, 1.0, 0.8), 0.04))

        # (1) 사각형 + (2) 전면 에지 + (3) 중심점
        if box is not None:
            arr.markers.append(self.mk_box(frame_id, stamp, 2, box, rgba(1.0, 0.3, 0.2, 0.9), 0.04))

            x_min = float(np.min(box[:, 0]))
            y_min = float(np.min(box[:, 1])); y_max = float(np.max(box[:, 1]))
            arr.markers.append(self.mk_front_edge(frame_id, stamp, 3, x_min, y_min, y_max, rgba(1.0, 1.0, 0.2, 0.9), 0.04))

            cx, cy = np.mean(box[:, 0]), np.mean(box[:, 1])
            arr.markers.append(self.mk_center(frame_id, stamp, 4, float(cx), float(cy), rgba(1.0, 0.6, 0.0, 0.9), 0.08))

        # (4) 텍스트(거리/상태)
        color = rgba(0.2, 1.0, 0.2, 0.9) if clear else rgba(1.0, 0.2, 0.2, 0.9)
        arr.markers.append(self.mk_text(frame_id, stamp, 5, 0.0, -0.4, f"d_front={d_front:.2f} m | clear={clear}", color, 0.15))

        self.pub_mark.publish(arr)

    # ---- Marker 빌더: RViz 시각화 요소 개별 생성 ----

    def mk_points(self, frame_id, stamp, mid, pts, color, scale):
        """
        POINTS Marker 생성.
        - 많은 점을 한 번에 표시할 때 유용.
        - geometry_msgs/Point 리스트에 x,y,z 할당(여기선 z=0).
        - lifetime으로 잔상 지속시간을 설정(여기선 0.3초).
        """
        m = Marker()
        m.header.frame_id = frame_id; m.header.stamp = stamp
        m.ns = self.marker_ns; m.id = mid; m.action = Marker.ADD
        m.type = Marker.POINTS
        m.scale.x = scale; m.scale.y = scale
        m.color = color
        for x, y in pts:
            p = Point(); p.x = float(x); p.y = float(y); p.z = 0.0
            m.points.append(p)
        m.lifetime = rclpy.duration.Duration(seconds=0.3).to_msg()
        return m

    def mk_box(self, frame_id, stamp, mid, box, color, width):
        """
        LINE_STRIP Marker로 사각형 윤곽을 그린다.
        - boxPoints로 받은 4점의 폐곡선을 만들기 위해 첫 점을 다시 붙인다.
        - scale.x는 선 두께.
        """
        m = Marker()
        m.header.frame_id = frame_id; m.header.stamp = stamp
        m.ns = self.marker_ns; m.id = mid; m.action = Marker.ADD
        m.type = Marker.LINE_STRIP
        m.scale.x = width
        m.color = color
        pts = np.vstack([box, box[0]])  # 폐곡선
        for x, y in pts:
            p = Point(); p.x = float(x); p.y = float(y); p.z = 0.0
            m.points.append(p)
        m.lifetime = rclpy.duration.Duration(seconds=0.3).to_msg()
        return m

    def mk_front_edge(self, frame_id, stamp, mid, x, y_min, y_max, color, width):
        """
        LINE_LIST Marker로 전면 에지 한 줄을 그린다.
        - 전면 에지는 x = min(box[:,0]) 수직선으로 표시.
        - LINE_LIST는 점 2개가 한 쌍으로 1개의 선을 이룸.
        """
        m = Marker()
        m.header.frame_id = frame_id; m.header.stamp = stamp
        m.ns = self.marker_ns; m.id = mid; m.action = Marker.ADD
        m.type = Marker.LINE_LIST
        m.scale.x = width
        m.color = color
        p1 = Point(); p1.x = x; p1.y = y_min; p1.z = 0.0
        p2 = Point(); p2.x = x; p2.y = y_max; p2.z = 0.0
        m.points = [p1, p2]
        m.lifetime = rclpy.duration.Duration(seconds=0.3).to_msg()
        return m

    def mk_center(self, frame_id, stamp, mid, x, y, color, scale):
        """
        SPHERE Marker로 사각형 중심점을 표시.
        - scale(x,y,z): 구의 지름.
        """
        m = Marker()
        m.header.frame_id = frame_id; m.header.stamp = stamp
        m.ns = self.marker_ns; m.id = mid; m.action = Marker.ADD
        m.type = Marker.SPHERE
        m.scale.x = scale; m.scale.y = scale; m.scale.z = scale
        m.color = color
        m.pose.position.x = x; m.pose.position.y = y; m.pose.position.z = 0.0
        m.lifetime = rclpy.duration.Duration(seconds=0.3).to_msg()
        return m

    def mk_text(self, frame_id, stamp, mid, x, y, text, color, scale):
        """
        TEXT_VIEW_FACING Marker로 텍스트(거리/판정)를 로봇 근처에 표시.
        - scale.z: 글자 높이
        - pose.position: 텍스트 기준점(바닥 z=0)
        """
        m = Marker()
        m.header.frame_id = frame_id; m.header.stamp = stamp
        m.ns = self.marker_ns; m.id = mid; m.action = Marker.ADD
        m.type = Marker.TEXT_VIEW_FACING
        m.scale.z = scale
        m.color = color
        m.pose.position.x = x; m.pose.position.y = y; m.pose.position.z = 0.0
        m.text = text
        m.lifetime = rclpy.duration.Duration(seconds=0.3).to_msg()
        return m


def rgba(r, g, b, a):
    """
    std_msgs/ColorRGBA 편의생성기.
    입력 범위: 0~1
    """
    c = ColorRGBA(); c.r = r; c.g = g; c.b = b; c.a = a
    return c


def main():
    """
    ROS2 엔트리 포인트.
    - rclpy.init() / spin(node) / shutdown() 표준 패턴.
    """
    rclpy.init()
    node = RectStopGoNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__=="__main__":
    main()