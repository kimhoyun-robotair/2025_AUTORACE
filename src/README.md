# 2025 스케일카 자율주행 경진대회
- 국민대학교를 포함한 미래자동차 컨소시엄에 참여한 학교들이 공동으로 주최하는 2025 스케일카 자율주행 경진대회 출전을 위한 리포지터리
- 본 대회는 총 9개의 미션을 구현해야하며, 미션은 다음과 같다. 
  - Lane Detection
  - 라바콘 피하기
  - 카메라를 활용하지 못하는 터널에서 주행
  - 평행 주차
  - 마커 기반으로 갈림길 탐색
  - 색상 기반으로 감속
  - 횡단보도 및 차단기 인식 및 정지 등 미션 수행
  - 회전교차로에서 장애물 회피 수행하며 주행

- 대회용으로 주어진 차량은 다음과 같은 구성을 가진다.
  - ackermann 타입의 차량
  - 2D LiDAR, IMU, Fisheye Camera (wideangle)

## 팀원 및 역할 분담
| **팀원 이름** | **담당하는 역할** |
| --- | --- |
| **임준형** | 팀장, Localization (VIO) |
| **김호윤** | Simulation 및 코드 통합 |
| **이민형** | 제어 (MPC) |
| **권준하** | ROS2 공부 및 연습 |

## 소프트웨어 스택 구현
- 기본적으로 Ubuntu 22.04 LTS와 ROS2 Humble을 사용하여 소프트웨어 스택을 구현한다.
- 소프트웨어 스택은 다음 내용들로 구성된다.
  - **VIO (Visual - Inertial Odometry)**
  - **MPC based Control**
  - **Simulation with Gazebo Harmonic, Gazebo Classic**
  - TBD

## 자체적인 시뮬레이션 구현
- Gazebo를 활용해서 자체적인 시뮬레이션을 구현
- ROS2 Humble과 Gazebo Harmonic, 그리고 Gazebo Classic을 통해서 자체적인 시뮬레이션 구현
- **Gazebo Harmonic에서는 Wideangle Camera의 사용이 제한됨 -> 사용하는 것을 추천하지 않음**
- **Ackermann 플러그인과 Wideangle Camera 시뮬레이션이 가능한 Gazebo Classic 사용을 강하게 추천함**

## 사용법
```bash
$ cd
$ git clone https://github.com/kimhoyun-robotair/2025_AUTORACE.git
$ cd 2025_AUTORACE
$ colcon build
$ source install/setup.bash
```

### Autorace Sim Harmonic (Not Recommended)
- 스케일카 자율주행 경진대회 출전을 위한 시뮬레이션 구현
- 다만, 본 시뮬레이션은 Migration 이후인 **Gazebo Harmonic**을 사용하였다.
- 사용법은 워크스페이스의 `src` 디렉터리 안에 있는 `autorace_sim` 패키지 내부의 `READMD.md` 파일을 참고할 것.

### Autorace Sim Classic (Highly Recommended)
- 스케일카 자율주행 경진대회 출전을 위한 시뮬레이션 구현
- 다만, 본 시뮬레이션은 **Gazebo Classi**c을 사용하였으며, 해당 버젼 사용을 강력하게 추천한다
- 사용법은 워크스페이스의 `src` 디렉터리 안에 있는 `autorace_sim_classic` 패키지의 내부의 `README.md` 파일을 참고할 것