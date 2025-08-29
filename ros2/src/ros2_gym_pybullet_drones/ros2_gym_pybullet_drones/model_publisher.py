import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose, Point, Quaternion

from ament_index_python.packages import get_package_share_directory
import os

class ModelPublisher(Node):
    def __init__(self):
        super().__init__('model_publisher')

        self.marker_pub = self.create_publisher(Marker, 'robot_model', 10)
        self.obs_sub = self.create_subscription(Float32MultiArray, 'obs', self.obs_callback, 10)

        self.mesh_path = os.path.join(
            get_package_share_directory('ros2_gym_pybullet_drones'),
            'meshes', 'yunque.dae'
        )
        self.frame_id = 'ground_link'

    def obs_callback(self, msg: Float32MultiArray):
        # Extract state values from the Float32MultiArray
        data = msg.data
        # Assuming the array structure: [pos(3), quat(4), rpy(3), vel(3), ang_v(3), action(3)]
        if len(data) < 7:
            self.get_logger().warn('Received array too short to extract pose information')
            return

        # Extract position (first 3 elements) and quaternion (next 4 elements)
        pos = data[0:3]
        quat = data[3:7]

        # Create a Pose message from extracted data
        pose = Pose()
        pose.position = Point(x=pos[0], y=pos[1], z=pos[2])
        pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])

        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'robot_model'
        marker.id = 0
        marker.type = Marker.MESH_RESOURCE
        marker.action = Marker.ADD

        marker.pose = pose

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