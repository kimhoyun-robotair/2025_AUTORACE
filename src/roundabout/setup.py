from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'roundabout'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # ament 패키지 인덱스 등록
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        # package.xml 설치
        ('share/' + package_name, ['package.xml']),
        # config / launch 파일들 (있다면) 함께 설치
        ('share/' + package_name + '/config', glob(os.path.join('config', '*.yaml'))),
        ('share/' + package_name + '/launch', glob(os.path.join('launch', '*.py'))),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'opencv-python',
    ],
    zip_safe=True,
    maintainer='jun',
    maintainer_email='gogownsud@gmail.com',
    description='Roundabout mission nodes (BEV lane extraction + roundabout FSM)',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # BEV 이미지 → /lane_pose
            'bev_lane_node = roundabout.bev_lane_node:main',
            # /scan, /imu, /lane_pose → /cmd_vel (FSM)
            'roundabout_node = roundabout.roundabout_node:main',
        ],
    },
)

