import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():

    pkg_autorace_sim = get_package_share_directory('autorace_sim')
    gazebo_models_path, ignore_last_dir = os.path.split(pkg_autorace_sim)
    os.environ["GZ_SIM_RESOURCE_PATH"] += os.pathsep + gazebo_models_path

    world_arg = DeclareLaunchArgument(
        'world', default_value='autoracemap.sdf',
        description='Name of the Gazebo world file to load'
    )

    model_arg = DeclareLaunchArgument(
        'model', default_value='autorace.urdf',
        description='Name of the URDF description to load'
    )

    x_arg = DeclareLaunchArgument(
        'x', default_value='-4.5',
        description='x coordinate of spawned robot'
    )

    y_arg = DeclareLaunchArgument(
        'y', default_value='0.5',
        description='y coordinate of spawned robot'
    )

    yaw_arg = DeclareLaunchArgument(
        'yaw', default_value='-1.5707',
        description='yaw angle of spawned robot'
    )

    sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='True',
        description='Flag to enable use_sim_time'
    )

    # Define the path to your URDF or Xacro file
    urdf_file_path = PathJoinSubstitution([
        pkg_autorace_sim,  # Replace with your package name
        "robot_description",
        LaunchConfiguration('model')  # Replace with your URDF or Xacro file
    ])

    world_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_autorace_sim, 'launch', 'world.launch.py'),
        ),
        launch_arguments={
        'world': LaunchConfiguration('world'),
        }.items()
    )

    # Spawn the URDF model using the `/world/<world_name>/create` service
    spawn_urdf_node = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-name", "physical_ai_bot",
            "-topic", "robot_description",
            "-x", LaunchConfiguration('x'), "-y", LaunchConfiguration('y'), "-z", "0.2", "-Y", LaunchConfiguration('yaw')  # Initial spawn position
        ],
        output="screen",
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
        ]
    )

    # Node to bridge /cmd_vel and /odom
    gz_bridge_node = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist",
            "/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry",
            "/joint_states@sensor_msgs/msg/JointState@gz.msgs.Model",
            "/tf@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V",
            # Sensor Module in Mast Topic
            "/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan",
            "/camera/image@sensor_msgs/msg/Image@gz.msgs.Image",
            "/camera/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo",
            "/imu@sensor_msgs/msg/Imu@gz.msgs.IMU",      
        ],
        output="screen",
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
        ],
        # remappings=[
            # ('/mast_imu', '/imu'),
            # ('/mast_scan', '/scan'), 
        # ]
    )

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            # xacro 결과를 "문자열"로 고정
            'robot_description': ParameterValue(
                Command(['xacro', ' ', urdf_file_path]),
                value_type=str
            ),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static')
        ]
    )

    launchDescriptionObject = LaunchDescription()

    launchDescriptionObject.add_action(world_arg)
    launchDescriptionObject.add_action(model_arg)
    launchDescriptionObject.add_action(x_arg)
    launchDescriptionObject.add_action(y_arg)
    launchDescriptionObject.add_action(yaw_arg)
    launchDescriptionObject.add_action(sim_time_arg)
    launchDescriptionObject.add_action(world_launch)
    launchDescriptionObject.add_action(spawn_urdf_node)
    launchDescriptionObject.add_action(gz_bridge_node)
    launchDescriptionObject.add_action(robot_state_publisher_node)

    return launchDescriptionObject