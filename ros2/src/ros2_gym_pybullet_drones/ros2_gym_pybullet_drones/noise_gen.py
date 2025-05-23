import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class DepthNoiseNode(Node):
    def __init__(self):
        super().__init__('depth_noise_node')
        self.subscription = self.create_subscription(
            Image, 'depth_image', self.depth_callback, 10)
        self.publisher = self.create_publisher(Image, 'depth_noisy', 10)
        self.bridge = CvBridge()
        self.L = 0.1  # Near clipping distance
        self.farVal = 1000  # Far clipping distance

    def depth_callback(self, msg):
        depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='64FC1')

        # Apply denormalization and noise in one step
        noisy_depth = self.add_depth_noise(depth_image, fov_horizontal=90, fov_vertical=90)

        # Convert back to ROS Image message and publish
        noisy_msg = self.bridge.cv2_to_imgmsg(noisy_depth, encoding='64FC1')
        self.publisher.publish(noisy_msg)

    def add_depth_noise(self, depth_map, fov_vertical=90, fov_horizontal=90):
        h, w = depth_map.shape
        fx = w / (2 * np.tan(np.radians(fov_horizontal / 2)))
        fy = h / (2 * np.tan(np.radians(fov_vertical / 2)))

        valid_mask = depth_map > 0  # Only apply noise to valid depth values

        # Compute axial noise vectorized
        axial_noise = np.random.normal(0, self.axial_noise(depth_map), depth_map.shape)
        
        # Apply axial noise only to valid depth pixels
        noisy_depth = np.where(valid_mask, depth_map + axial_noise, depth_map)

        # Compute lateral noise vectorized
        lateral_noise = self.lateral_noise(depth_map, fx)

        # Compute new x, y coordinates
        x_indices, y_indices = np.meshgrid(np.arange(w), np.arange(h))
        x_noisy = np.clip(x_indices + lateral_noise[0], 0, w - 1).astype(np.int32)
        y_noisy = np.clip(y_indices + lateral_noise[1], 0, h - 1).astype(np.int32)

        # Apply lateral noise without modifying original array in-place
        noisy_depth_shifted = np.copy(noisy_depth)
        noisy_depth_shifted[y_noisy, x_noisy] = noisy_depth[y_indices, x_indices]

        return noisy_depth_shifted

    def axial_noise(self, z):
        """ Vectorized axial noise computation """
        sigma_z = 0.001063 + 0.0007278 * z + 0.003949 * z * z
        # sigma_z = 0.0012 + 0.0019*(z-0.4)**2
        return np.where(z > 0, sigma_z, 0)  # Apply noise only to valid depth pixels

    def lateral_noise(self, z, fx=585):
        """ Vectorized lateral noise computation """
        # sigma_L = 0.0016
        sigma_L = 0.00
        noise_x = np.random.normal(0, sigma_L, z.shape)
        noise_y = np.random.normal(0, sigma_L, z.shape)
        return np.stack((noise_x, noise_y), axis=0)  # Shape (2, H, W)

def main(args=None):
    rclpy.init(args=args)
    node = DepthNoiseNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
