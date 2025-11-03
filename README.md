# 2025 스케일카 자율주행 경진대회 시뮬레이션
- 스케일카 시뮬레이션 사용법에 대해서 정리해놓는 README.md

## 기초적인 사용법
### 일단 필요한 것들
- ROS2 Humbel
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