from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'autorace_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
	    (os.path.join('share', package_name), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kimhoyun',
    maintainer_email='suberkut76@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'lane_detection = autorace_detection.lane_detectoin:main',
            'stopline_detection = autorace_detection.stopline_detection:main',
            'red_blue_detector = autorace_detection.red_blue_detector:main',
            'state_machine = autorace_detection.state_machine:main',
            'lane_controller = autorace_detection.lane_controller:main',
            'color_controller = autorace_detection.color_controller:main',
            'stopline_controller = autorace_detection.stopline_controller:main',
        ],
    },
)
