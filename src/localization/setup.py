from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'localization'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(include=[package_name]),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # install config & launch files
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jun',
    maintainer_email='gogownsud@gmail.com',
    description='BEV image node and static TF node for autorace localization.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bev_node = localization.bev_node:main',
            'static_tf_node = localization.static_tf_node:main',
        ],
    },
)

