from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'maskdino_ros_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        (os.path.join('share', package_name, 'params'), glob('params/*')),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Nshenoy344',
    maintainer_email="nshenoy344@gmail.com",
    description='get detections from maskdino',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'maskdino_ros = maskdino_ros_pkg.maskdino_ros:main'
        ],
    },
)
