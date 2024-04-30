from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            # Launch the node
            Node(
                package="kinect2_map",
                executable="kinect2_map_node",
                name="kinect2_map_node",
                output="screen"
            )
        ]
    )
