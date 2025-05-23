#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped, PoseStamped, Point
from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from px4_msgs.msg import VehicleOdometry
from visualization_msgs.msg import Marker
import tf2_ros
import tf_transformations  # Use this for quaternion calculations
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

class DummyTFAndPathPublisher(Node):
    def __init__(self):
        super().__init__('dummy_tf_and_path_publisher')
        self.odom = VehicleOdometry()
        # TF broadcaster for the dummy transform
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Publisher for the path (waypoints)
        self.path_pub = self.create_publisher(Path, 'waypoints', 10)
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,  # Match the publisher's reliability policy
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,  # Match the publisher's durability policy
            depth=10  # Adjust the depth as needed
        )
        self.odom_sub = self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self.get_odom_callback, qos_profile)
        # Publisher for the marker visualization
        self.marker_pub = self.create_publisher(Marker, 'visualization_marker', 10)

        # Timer to periodically publish both messages
        self.timer = self.create_timer(0.1, self.timer_callback)

    def get_odom_callback(self, msg):
        print("odometry callback")
        self.odom = msg
        self.broadcast_transform(self.odom.position[0], self.odom.position[1], self.odom.position[2], self.odom.q[0], self.odom.q[1], self.odom.q[2], self.odom.q[3], "base_link", "ground_link")

    def timer_callback(self):
        self.publish_tf()
        self.publish_path()
        self.publish_marker()

    def publish_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'ground_link'

        # Zero translation
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0

        # Apply 180-degree (pi radians) yaw rotation
        q = tf_transformations.quaternion_from_euler(0, 0, 0)  # Roll=0, Pitch=0, Yaw=180°

        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(t)
        self.get_logger().debug('Broadcasting corrected transform from "map" to "ground_link"')

    def broadcast_transform(self, x, y, z, qw, qx, qy, qz, frame_name, parent_name):
        # Convert position from NED (PX4) to ENU (ROS):
        # NED: +X (North), +Y (East), +Z (Down)
        # ENU: +X (East), +Y (North), +Z (Up)
        # So we convert:
        #   x_enu = y_ned
        #   y_enu = x_ned
        #   z_enu = -z_ned
        drone_position = [float(x), -float(y), -float(z)]

        # Original PX4 quaternion: [qw, qx, qy, qz]
        # Reformat to [x, y, z, w] for tf
        q_ned = [float(qx), float(qy), float(qz), float(qw)]

        # Convert rotation from NED to ENU frame:
        # Apply 90° yaw + 180° roll rotation to map axes correctly
        q_rot = tf_transformations.quaternion_from_euler(0, 0, 1.5708)  # roll=π, yaw=π/2

        # Combined rotation: q_enu = q_rot * q_ned
        q_nwu = tf_transformations.quaternion_multiply(q_rot, q_ned)
        # Prepare the TransformStamped message
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent_name
        t.child_frame_id = frame_name

        t.transform.translation.x = drone_position[0]
        t.transform.translation.y = drone_position[1]
        t.transform.translation.z = drone_position[2]

        t.transform.rotation.x = q_nwu[0]
        t.transform.rotation.y = q_nwu[1]
        t.transform.rotation.z = q_nwu[2]
        t.transform.rotation.w = q_nwu[3]

        self.tf_broadcaster.sendTransform(t)

    def publish_path(self):
        now = self.get_clock().now().to_msg()
        path = Path()
        path.header.stamp = now
        path.header.frame_id = 'ground_link'  # Change to ground_link

        # Create a single PoseStamped for the dummy waypoint
        pose = PoseStamped()
        pose.header.stamp = now
        pose.header.frame_id = 'ground_link'  # Change to ground_link
        pose.pose.position.x = 50.0
        pose.pose.position.y = 0.0
        pose.pose.position.z = 3.0
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 1.0

        # Append the dummy waypoint to the Path
        path.poses.append(pose)

        # Publish the Path message
        self.path_pub.publish(path)
        self.get_logger().debug('Publishing path with a waypoint at (10,10,10) in ground_link')

    def publish_marker(self):
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = 'ground_link'  # Match frame to ground_link
        marker.ns = "waypoint_marker"
        marker.id = 0
        marker.type = Marker.SPHERE  # Sphere shape
        marker.action = Marker.ADD

        # Set marker position to the waypoint
        marker.pose.position.x = -20.0
        marker.pose.position.y = -20.0
        marker.pose.position.z = 3.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        # Set scale (size) of the marker
        marker.scale.x = 1.0  # Radius of the sphere
        marker.scale.y = 1.0
        marker.scale.z = 1.0

        # Set color of the marker (RGBA)
        marker.color.r = 1.0  # Red
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0  # Full opacity

        # Publish the marker
        self.marker_pub.publish(marker)
        self.get_logger().debug("Publishing marker at (10,10,10) in ground_link")

def main(args=None):
    rclpy.init(args=args)
    node = DummyTFAndPathPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
