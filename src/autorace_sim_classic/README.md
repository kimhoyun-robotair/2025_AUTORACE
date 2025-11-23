# 2025 스케일카 자율주행 경진대회 시뮬레이션 with Classic (Highly Recommended)
- 스케일카 시뮬레이션 사용법에 대해서 정리해놓는 README.md
- 본 시뮬레이션은 Gazebo Classic을 사용하였다.
- Ackeramnn Drive 플러그인, 2D LiDAR, IMU, Fisheye Camera 시뮬레이션을 지원한다.

## 기초적인 사용법
### 일단 필요한 것들
- ROS2 Humble
- Gazebo Classic
- gazebo_ros 패키지
- slam-toolbox (그냥 설치만 되어있으면 됨)

### Gazebo Classic 설치방법
- Gazebo Harmonic을 지우고 나서 Gazebo Classic 설치하기 (**추천**)
```bash
$ cd
$ sudo apt remove gz-harmonic && sudo apt autoremove
$ sudo apt remove ros-${DISTRO}-ros-gz && sudo apt autoremove
$ sudo apt update && sudo apt upgrade
$ sudo apt install gazebo11 libgazebo11-dev ros-${DISTRO}-gazebo-ros ros-${DISTRO}-gazebo-plugins

# For Test
$ gazebo
```
- Gazebo Harmonic과 Gazebo Classic을 동시에 설치하고 같이 사용하기
```bash
$ sudo add-apt-repository ppa:openrobotics/gazebo11-gz-cli
$ sudo apt update
$ sudo apt-get install gazebo11
$ sudo apt install libgazebo11-dev ros-${DISTRO}-gazebo-ros ros-${DISTRO}-gazebo-plugins

# For Test
$ gazebo
```

### 사용법
- 먼저, `git clone`을 통해서 워크스페이스 전체를 다운로드 해야할 필요가 있다.
```bash
$ cd
$ git clone -b hyk-test https://github.com/kimhoyun-robotair/2025_AUTORACE.git
```

- 그 다음으로, 원활한 시뮬레이션을 위해 Gazebo의 모델 경로를 인식시켜줄 필요가 있다.
- `autorace_sim_classic` 내부의 `model` 디렉터리 안에 있는 파일들을 다음 경로로 복사하자.
  - `~/.gazebo/models`
  - `models` 폴더가 없다면 직접 만든 다음에 복사하면 된다.

- 이후 빌드를 하고, 시뮬레이션을 사용하면 된다.
```bash
$ cd
$ cd 2025_AUTORACE
$ colcon build
$ source install/setup.bash
$ ros2 launch autorace_sim_classic spawn_robot.launch.py
```
- 시뮬레이션에 사용되는 map을 바꾸고 싶다면, 우선 `autorace_sim_classic`의 `world` 파일 안에 커스텀 맵을 추가하고,
- 이후 `spawn_robot.launch.py` 파일의 다음 부분을 변경하면 된다.
```python
    world_path=os.path.join(pkg_share, 'world/demomap/model.sdf')
```

- 만약 `slam_toolbox`를 사용하고 싶다면 다른 터미널을 열어서 다음과 같이 사용하면 된다.
```bash
$ cd /path/to/your/autorace_workspace/
$ source install/setup.bash
$ ros2 launch autorace_sim slamtoolbox.launch.py
```
- `slam_toolbox`의 설정을 조정하고 싶다면, `src/autorace_sim/config` 내에 있는 `YAML` 파일의 파라미터를 조정하자.

- 만약 `keyboard_teleop`을 사용하고 싶다면 시뮬레이션을 연 상태에서 터미널을 하나 더 열고, 다음과 같이 입력하면 된다.
```bash
$ ros2 run teleop_twist_keyboard teleop_twist_keyboard
```
  
### 문의사항
- 안될 경우 `suberkut76@gmail.com` 혹은 카톡방, Github Issue 등으로 문의할 것