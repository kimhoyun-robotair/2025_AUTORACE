from setuptools import setup
import os
from glob import glob

package_name = 'razor_imu_9dof_ros2'

setup(
    name=package_name,
    version='1.3.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Launch 파일 설치
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # Config 파일 설치
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='User',
    maintainer_email='user@todo.todo',
    description='Razor IMU 9DOF Driver for ROS 2',
    license='BSD',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'imu_node = razor_imu_9dof_ros2.imu_node:main',
        ],
    },
)
