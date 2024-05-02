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
                output="screen",
                parameters=[
                    {
                        "crop_scene": True,
                        "prune_unreachable": True,
                        "fill_back": True,
                        "octree_resolution": 0.01,
                        "octree_output_path": "/home/kafka/octree.bt",
                        "octree_augmented_output_path": "/home/kafka/octree_augmented.bt",
                        "intarray_output_path": "/home/kafka/intarray.bin",
                        "x_rotation": 262.0,
                        "y_rotation": 0.0,
                        "z_rotation": 0.0,
                        "x_translation": 0.66,
                        "y_translation": -0.85,
                        "z_translation": 0.38,
                        "scene_x_max": 1.27,
                        "scene_y_max": 0.63,
                        "scene_z_max": 0.63,
                        "fill_y_threshold": 0.20,
                        "arm_base_x": 0.61,
                        "arm_base_y": 0.00,
                        "arm_base_z": 0.00,
                        "arm_max_reach": 0.50,
                        "start_x": 0.45,
                        "start_y": 0.35,
                        "start_z": 0.20,
                        "end_x": 0.85,
                        "end_y": 0.35,
                        "end_z": 0.20
                    }
                ]
            )
        ]
    )
