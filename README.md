# 2025 스케일카 자율주행 경진대회 시뮬레이션
- 스케일카 시뮬레이션 사용법에 대해서 정리해놓는 README.md

## 기초적인 사용법
### 일단 필요한 것들
- ROS2 Humble
- Gazebo Harmonic
- ros_gz 패키지 (무조건 Harmonic에 맞춰서)
- slam-toolbox (그냥 설치만 되어있으면 됨)
- Nav2 (추후 추가 예정이므로 2025.11.03 기준 필요없음)

### 사용법
- 일단 `Gazebo Harmonic` 경로를 인식해줄 필요가 있음
```bash
$ export GZ_SIM_RESOURCE_PATH=~/path/to/your/autorace_workspace/src/autorace_sim/model
```
- 그 다음 `Gazebo` 열어버리면 된다.
```bash
$ cd /path/to/your/autorace_workspace/
$ colcon build
$ source install/setup.bash
$ ros2 launch autorace_sim spawn_robot.launch.py
```

### 문의사항
- 안될 경우 `suberkut76@gmail.com` 혹은 카톡방, Github Issue 등으로 문의할 것


###
###
## 여기부터 임준형 영역 Localization
전체 구조는 다음과 같다.
- bev_node
  - 입력 : /image_raw
  - 출력 : /image_bev
- static_tf_node
  - 입력 : camera_config.yaml, tf_config.yaml
  - 출력 : base_link -> camera, base_link -> BEV (Bird's Eye View)
- edge_node
  - 입력 : /image_bev
  - 출력 : /edge_cloud, /lane_width_hint
- mcl_matcher_node
  - 입력 : /edge_cloud, /lane_width_hint, /imu
  - 출력 : /odom_camera (Odometry Nav message)
- EKF_fuser
  - 입력 : /odom_camera, /imu
  - 출력 : /tf, /odom (map -> odom, odom -> base_link Odometry Nav message)
