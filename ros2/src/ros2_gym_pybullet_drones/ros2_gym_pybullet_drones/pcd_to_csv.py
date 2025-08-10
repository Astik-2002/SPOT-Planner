# #!/usr/bin/env python3

# import rclpy
# from rclpy.node import Node
# import sensor_msgs_py.point_cloud2 as pc2
# import csv
# import numpy as np
# from sensor_msgs.msg import PointCloud2
# import time
# import os
# from datetime import datetime

# import tf2_ros

# class PointCloudToCSV(Node):

#     def __init__(self):
#         super().__init__('point_cloud_to_csv')
#         self.subscription = self.create_subscription(
#             PointCloud2,
#             'pcd_gym_pybullet',
#             # '/camera/camera/depth/color/points',  # Replace with your actual topic
#             self.point_cloud_callback,
#             10)
#         self.subscription  # Prevent unused variable warning
#         self.counter = 0  # Counter to track the number of point clouds saved
#         current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#         base_path = '/home/astik/pcd_outputs/'
#         folder_name = current_time
#         self.full_path = os.path.join(base_path, folder_name)

#         # Check whether the specified path exists or not
#         isExist = os.path.exists(self.full_path)
#         if not isExist:
#             os.makedirs(self.full_path)

#     def point_cloud_callback(self, point_cloud):
#         # Convert PointCloud2 to an array
#         points_list = []

#         for point in pc2.read_points(point_cloud, skip_nans=True):
#             points_list.append([point[0], point[1], point[2]])  # X, Y, Z coordinates

#         # Convert to numpy array
#         points_array = np.array(points_list)

#         # Save to CSV with a unique name
#         file_name = os.path.join(self.full_path, f'pcd_{self.counter:04d}.csv')
#         with open(file_name, 'w') as file:
#             writer = csv.writer(file)
#             writer.writerow(["X", "Y", "Z"])  # Write the header
#             writer.writerows(points_array)

#         self.get_logger().info(f'Point cloud saved to {file_name}')

#         # Increment the counter for the next file
#         self.counter += 1

#         # Ensure the callback is invoked at 2 Hz
#         time.sleep(0.5)

# def main(args=None):
#     rclpy.init(args=args)
#     point_cloud_to_csv_node = PointCloudToCSV()
#     rclpy.spin(point_cloud_to_csv_node)

#     # Destroy the node explicitly
#     point_cloud_to_csv_node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import sensor_msgs_py.point_cloud2 as pc2
import csv
import numpy as np
from sensor_msgs.msg import PointCloud2
from custom_interface_gym.msg import BoundingBoxArray
import time
import os
from datetime import datetime

class SyncedLogger(Node):
    def __init__(self):
        super().__init__('synced_logger')

        self.pc_subscription = self.create_subscription(
            PointCloud2,
            'pcd_gym_pybullet',
            self.point_cloud_callback,
            10)

        self.bbox_subscription = self.create_subscription(
            BoundingBoxArray,
            'dynamic_obs_state',  
            self.bbox_callback,
            10)

        self.latest_pointcloud = None
        self.latest_bbox_array = None

        self.counter = 0
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_path = '/home/astik/pcd_outputs/'
        self.full_path = os.path.join(base_path, current_time)
        os.makedirs(self.full_path, exist_ok=True)

    def point_cloud_callback(self, msg):
        self.latest_pointcloud = msg
        self.try_save_frame()

    def bbox_callback(self, msg):
        self.latest_bbox_array = msg
        self.try_save_frame()

    def try_save_frame(self):
        if self.latest_pointcloud is None or self.latest_bbox_array is None:
            return

        pc_time = self.latest_pointcloud.header.stamp.sec + self.latest_pointcloud.header.stamp.nanosec * 1e-9
        bbox_time = self.latest_bbox_array.header.stamp.sec + self.latest_bbox_array.header.stamp.nanosec * 1e-9
        time_diff = abs(pc_time - bbox_time)

        if time_diff < 0.05:
            # Create subfolder for this frame
            frame_folder_name = f'{self.counter:04d}'
            frame_folder_path = os.path.join(self.full_path, frame_folder_name)
            os.makedirs(frame_folder_path, exist_ok=True)

            # Save point cloud
            pc_filename = os.path.join(frame_folder_path, f'pcd_{self.counter:04d}.csv')
            points = [
                [pt[0], pt[1], pt[2]]
                for pt in pc2.read_points(self.latest_pointcloud, skip_nans=True)
            ]
            with open(pc_filename, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(["X", "Y", "Z"])
                writer.writerows(points)
            self.get_logger().info(f"Saved point cloud: {pc_filename}")

            # Save bounding boxes
            bbox_filename = os.path.join(frame_folder_path, f'bbox_array_{self.counter:04d}.csv')
            with open(bbox_filename, 'w') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Center_X", "Center_Y", "Center_Z",
                    "Height", "Length", "Width",
                    "Velocity_X", "Velocity_Y", "Velocity_Z"
                ])
                for box in self.latest_bbox_array.boxes:
                    writer.writerow([
                        box.center_x, box.center_y, box.center_z,
                        box.height, box.length, box.width,
                        box.velocity_x, box.velocity_y, box.velocity_z
                    ])
            self.get_logger().info(f"Saved bounding boxes: {bbox_filename}")

            self.latest_pointcloud = None
            self.latest_bbox_array = None
            self.counter += 1

def main(args=None):
    rclpy.init(args=args)
    node = SyncedLogger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
