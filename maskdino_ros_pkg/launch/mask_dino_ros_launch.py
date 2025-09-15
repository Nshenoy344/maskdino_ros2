from launch import LaunchDescription
from launch_ros.actions import Node
import os

from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    config = os.path.join(
      get_package_share_directory('maskdino_ros_pkg'),
      'params',
      'params.yaml'
      )

    return LaunchDescription([
        Node(
            package='maskdino_ros_pkg',
            namespace='',
            executable='maskdino_ros',
            name='maskdino_ros2',
            parameters=[config]
        )
    ])