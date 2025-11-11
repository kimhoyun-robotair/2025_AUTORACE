# 2025 스케일카 자율주행 경진대회 시뮬레이션
- 스케일카 시뮬레이션 사용법에 대해서 정리해놓는 README.md
- 다만, 본 시뮬레이션은 Migration 이후인 Gazebo Harmonic을 사용하였다.

## 기초적인 사용법
### 일단 필요한 것들
- ROS2 Humble
- Gazebo Harmonic
- ros_gz 패키지 (무조건 Harmonic에 맞춰서)
- slam-toolbox (그냥 설치만 되어있으면 됨)
- Nav2 (추후 추가 예정이므로 2025.11.03 기준 필요없음)

### 사용법
- 먼저, `git clone`을 통해서 워크스페이스 전체를 다운로드 해야할 필요가 있다.
```bash
$ cd
$ git clone -b hyk-test https://github.com/kimhoyun-robotair/2025_AUTORACE.git
$ cd 2025_AUTORACE
$ colcon ubild
$ source install/setup.bash
```

- 이후 `Gazebo Harmonic` 경로를 인식해줄 필요가 있음
```bash
$ export GZ_SIM_RESOURCE_PATH=~/path/to/your/autorace_workspace/src/autorace_sim/model
```

- 그 다음 `Gazebo`를 열어버리면 된다.
```bash
$ cd /path/to/your/autorace_workspace/
$ colcon build
$ source install/setup.bash
$ ros2 launch autorace_sim spawn_robot.launch.py
```

- 만약 `slam_toolbox`를 사용하고 싶다면 다른 터미널을 열어서 다음과 같이 사용하면 된다.
```bash
$ cd /path/to/your/autorace_workspace/
$ source install/setup.bash
$ ros2 launch autorace_sim slamtoolbox.launch.py
```
- `slam_toolbox`의 설정을 조정하고 싶다면, `src/autorace_sim/config` 내에 있는 `YAML` 파일의 파라미터를 조정하자.
  
### 문의사항
- 안될 경우 `suberkut76@gmail.com` 혹은 카톡방, Github Issue 등으로 문의할 것