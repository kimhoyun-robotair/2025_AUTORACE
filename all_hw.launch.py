import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    

    pkg_f1tenth = get_package_share_directory('f1tenth_stack')
    launch_vesc = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_f1tenth, 'launch', 'bringup_launch.py')
        )
    )


    pkg_lidar = get_package_share_directory('rplidar_ros') 
    launch_lidar = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_lidar, 'launch', 'rplidar_s1_launch.py') 
        )
    )
    
    
    pkg_cam = get_package_share_directory('localization') 
    launch_cam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_cam, 'launch', 'launch.py') 
        )
    )
    
    
    pkg_imu = get_package_share_directory('razor_imu_9dof_ros2') 
    launch_imu = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_imu, 'launch', 'razor.launch.py') 
        )
    )



    # === 3. 동시에 실행 ===
    return LaunchDescription([
        launch_vesc,
        launch_lidar,
        launch_cam , 
        launch_imu ,
    ])
