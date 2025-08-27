import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose

from ament_index_python.packages import get_package_share_directory
import os

class ModelPublisher(Node):
    def __init__(self):
        super().__init__('model_publisher')

        self.marker_pub = self.create_publisher(Marker, 'robot_model', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        self.mesh_path = os.path.join(
            get_package_share_directory('ros2_gym_pybullet_drones'),
            'meshes', 'yunque.dae'
        )
        self.frame_id = 'ground_link'

    def odom_callback(self, msg: Odometry):
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'robot_model'
        marker.id = 0
        marker.type = Marker.MESH_RESOURCE
        marker.action = Marker.ADD

        marker.pose = msg.pose.pose

        marker.scale.x = marker.scale.y = marker.scale.z = 1.0  # Adjust if needed
        marker.color.a = 1.0
        marker.color.r = marker.color.g = marker.color.b = 1.0

        marker.mesh_resource = f'file://{self.mesh_path}'
        marker.mesh_use_embedded_materials = True

        self.marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = ModelPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
