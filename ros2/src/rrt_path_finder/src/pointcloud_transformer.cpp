#include <memory>
#include <vector>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include <Eigen/Dense>

// Message types
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/point_field.hpp"
#include "px4_msgs/msg/vehicle_odometry.hpp"

// For iterating through PointCloud2 data
#include "sensor_msgs/point_cloud2_iterator.hpp"

// For transformations (quaternion, matrix, vector)
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2/LinearMath/Vector3.h"

// ---- PCL includes
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

class PointCloudTransformer : public rclcpp::Node
{
public:
  PointCloudTransformer()
  : Node("pointcloud_transformer")
  {
    // Set up the odometry subscriber.
    auto qos_odometry = rclcpp::QoS(rclcpp::KeepLast(1));
    qos_odometry.reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT);
    odom_sub_ = this->create_subscription<px4_msgs::msg::VehicleOdometry>(
      "/fmu/out/vehicle_odometry", qos_odometry,
      std::bind(&PointCloudTransformer::odom_callback, this, std::placeholders::_1));

    // Set up the point cloud subscriber.
    pc_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/depth_camera/points", 10,
      std::bind(&PointCloudTransformer::pc_callback, this, std::placeholders::_1));

    // Publisher for the transformed + filtered point cloud.
    // pc_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/transformed_points", 10);
    auto qos = rclcpp::QoS(rclcpp::KeepLast(100));
    qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
    pc_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/transformed_points", qos);

    // Initialize the fixed transformation constants (camera to body).
    R_camera_body_ = tf2::Matrix3x3(
      0, 0, 1,
      1, 0, 0,
      0, 1, 0);

    t_camera_body_ = tf2::Vector3(0.1, 0.0, -0.05);
  }

private:
  // Callback for odometry messages.
  void odom_callback(const px4_msgs::msg::VehicleOdometry::SharedPtr msg)
  {
    RCLCPP_WARN(this->get_logger(), "in odometry callback...");
    latest_odom_ = msg;
  }

  // Callback for point cloud messages.
  void pc_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    if (!latest_odom_) {
      RCLCPP_WARN(this->get_logger(), "Waiting for odometry...");
      return;
    }

    // --- Extract odometry data and compute the body-to-map transform ---
    // Adjust quaternion indexing if needed; for px4_msgs, q = [w, x, y, z].
    double q_w = latest_odom_->q[0];
    double q_x = latest_odom_->q[1];
    double q_y = -latest_odom_->q[2];
    double q_z = -latest_odom_->q[3];

    tf2::Quaternion q_tf(q_x, q_y, q_z, q_w);
    tf2::Matrix3x3 R_body_map(q_tf);

    tf2::Vector3 t_body_map(
      latest_odom_->position[0],
      -latest_odom_->position[1],
      -latest_odom_->position[2]
    );

    // --- Convert input ROS2 PointCloud2 -> iterate points, transform to map frame ---
    sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");

    // Create a PCL cloud to hold the transformed points
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_unfiltered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl_cloud_unfiltered->reserve(msg->width * msg->height);  // Reserve space if you like

    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
      // Original camera point
      tf2::Vector3 p_camera(*iter_x, *iter_y, *iter_z);

      // Example: If you wanted to incorporate camera->body transform
      // p_camera = R_camera_body_ * p_camera + t_camera_body_;

      // Body->map transform:
      tf2::Vector3 p_map = R_body_map * p_camera + t_body_map;

      // Push into the PCL container
      pcl::PointXYZ pcl_pt;
      pcl_pt.x = static_cast<float>(p_map.x());
      pcl_pt.y = static_cast<float>(p_map.y());
      pcl_pt.z = static_cast<float>(p_map.z());
      pcl_cloud_unfiltered->push_back(pcl_pt);
    }

    // --- Apply VoxelGrid filter with leaf size of 0.6 ---
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setInputCloud(pcl_cloud_unfiltered);
    voxel_filter.setLeafSize(0.6f, 0.6f, 0.6f);
    voxel_filter.filter(*pcl_cloud_filtered);

    // --- Convert filtered PCL cloud -> ROS2 PointCloud2 ---
    sensor_msgs::msg::PointCloud2 output;
    pcl::toROSMsg(*pcl_cloud_filtered, output);

    // Set the appropriate header/frame for ROS
    output.header.stamp = this->now();
    output.header.frame_id = "ground_link";  // or your desired map frame

    // Publish the transformed + filtered cloud
    pc_pub_->publish(output);
  }

  // --- Member variables ---
  rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pc_pub_;

  px4_msgs::msg::VehicleOdometry::SharedPtr latest_odom_;

  // Fixed transform from camera to body
  tf2::Matrix3x3 R_camera_body_;
  tf2::Vector3   t_camera_body_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PointCloudTransformer>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
  

// ---------------------------------------- Testing synchronised pointcloud and odometry ----------------------------------------------
// #include <memory>
// #include <vector>
// #include <string>

// #include "rclcpp/rclcpp.hpp"
// #include "sensor_msgs/msg/point_cloud2.hpp"
// #include "sensor_msgs/msg/point_field.hpp"
// #include "px4_msgs/msg/vehicle_odometry.hpp"

// // To iterate over PointCloud2 data
// #include "sensor_msgs/point_cloud2_iterator.hpp"

// // Eigen for vectorized math
// #include <Eigen/Dense>
// #include <Eigen/Geometry>

// class PointCloudTransformer : public rclcpp::Node
// {
// public:
//   PointCloudTransformer() : Node("pointcloud_transformer")
//   {
//     // --- Set up subscriptions and publisher ---

//     // Odometry subscriber (using BEST_EFFORT QoS and a queue size of 1)
//     auto qos_odometry = rclcpp::QoS(rclcpp::KeepLast(1));
//     qos_odometry.reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT);
//     odom_sub_ = this->create_subscription<px4_msgs::msg::VehicleOdometry>(
//       "/fmu/out/vehicle_odometry", qos_odometry,
//       std::bind(&PointCloudTransformer::odom_callback, this, std::placeholders::_1));

//     // PointCloud2 subscriber (queue size of 10)
//     pc_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
//       "/depth_camera/points", 10,
//       std::bind(&PointCloudTransformer::pc_callback, this, std::placeholders::_1));

//     // Publisher for the transformed point cloud.
//     pc_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/transformed_points", 10);

//     // --- Fixed transformation from camera to body ---
//     // Here we only use a fixed translation (adjust units as necessary)
//     t_camera_body_ = Eigen::Vector3f(0.1f, 0.0f, -0.05f);
//   }

// private:
//   // Callback for odometry messages.
//   void odom_callback(const px4_msgs::msg::VehicleOdometry::SharedPtr msg)
//   {
//     RCLCPP_WARN(this->get_logger(), "Received odometry message...");
//     latest_odom_ = msg;
//   }

//   // Callback for point cloud messages.
//   void pc_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
//   {
//     // Wait until we have an odometry message.
//     if (!latest_odom_) {
//       RCLCPP_WARN(this->get_logger(), "Waiting for odometry...");
//       return;
//     }

//     // --- Extract odometry data and build transformation ---
//     //
//     // The PX4 odometry message provides the quaternion as [w, x, y, z].
//     // Eigen::Quaternionf uses the (w, x, y, z) constructor.
//     double q_w = latest_odom_->q[0];
//     double q_x = latest_odom_->q[1];
//     double q_y = latest_odom_->q[2];
//     double q_z = latest_odom_->q[3];
//     Eigen::Quaternionf q(static_cast<float>(q_w),
//                          static_cast<float>(q_x),
//                          static_cast<float>(q_y),
//                          static_cast<float>(q_z));
//     Eigen::Matrix3f R_body_map = q.toRotationMatrix();

//     // The odometry message’s position is given as:
//     // position[0]: x, position[1]: y, position[2]: z.
//     // Note that the original code negates y and z.
//     Eigen::Vector3f t_body_map(static_cast<float>(latest_odom_->position[0]),
//                                -static_cast<float>(latest_odom_->position[1]),
//                                -static_cast<float>(latest_odom_->position[2]));

//     // Our overall transformation is:
//     //   p_map = R_body_map * (p_camera + t_camera_body_) + t_body_map
//     // which can be re-written as:
//     //   p_map = R_body_map * p_camera + (R_body_map * t_camera_body_ + t_body_map)
//     Eigen::Vector3f t_total = R_body_map * t_camera_body_ + t_body_map;

//     // Build the 4x4 homogeneous transformation matrix.
//     Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
//     T.block<3,3>(0,0) = R_body_map;
//     T.block<3,1>(0,3) = t_total;

//     // --- Process the point cloud in a batch using Eigen ---

//     // Determine the number of points.
//     size_t num_points = msg->width * msg->height;
//     if (num_points == 0) {
//       RCLCPP_WARN(this->get_logger(), "Received an empty point cloud.");
//       return;
//     }

//     // Create an Eigen matrix to store the points (3 rows x N columns).
//     Eigen::MatrixXf points(3, num_points);

//     // Use PointCloud2 iterators to extract the x, y, and z fields.
//     sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
//     sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
//     sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");

//     for (size_t i = 0; i < num_points; ++i, ++iter_x, ++iter_y, ++iter_z) {
//       points(0, i) = *iter_x;
//       points(1, i) = *iter_y;
//       points(2, i) = *iter_z;
//     }

//     // Convert the points to homogeneous coordinates (4 x N).
//     Eigen::MatrixXf points_homogeneous(4, num_points);
//     points_homogeneous.topRows(3) = points;
//     points_homogeneous.row(3).setOnes();

//     // Apply the transformation in one shot.
//     Eigen::MatrixXf transformed_points = T * points_homogeneous;  // (4 x num_points)

//     // --- Create the output PointCloud2 message ---
//     sensor_msgs::msg::PointCloud2 output;
//     output.header.stamp = this->now();
//     output.header.frame_id = "ground_link";  // Adjust as necessary.
//     output.height = 1;
//     output.width = num_points;
//     output.is_bigendian = false;
//     output.is_dense = true;
//     output.point_step = 12;  // 3 floats (x,y,z) * 4 bytes each.
//     output.row_step = output.point_step * output.width;

//     // Define the fields: x, y, and z.
//     output.fields.resize(3);
//     output.fields[0].name   = "x";
//     output.fields[0].offset = 0;
//     output.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
//     output.fields[0].count  = 1;

//     output.fields[1].name   = "y";
//     output.fields[1].offset = 4;
//     output.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
//     output.fields[1].count  = 1;

//     output.fields[2].name   = "z";
//     output.fields[2].offset = 8;
//     output.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
//     output.fields[2].count  = 1;

//     // Allocate space for the point data.
//     output.data.resize(output.row_step * output.height);

//     // Copy the transformed points (only the top 3 rows) into the output data buffer.
//     for (size_t i = 0; i < num_points; ++i) {
//       float* ptr = reinterpret_cast<float*>(&output.data[i * output.point_step]);
//       ptr[0] = transformed_points(0, i);
//       ptr[1] = transformed_points(1, i);
//       ptr[2] = transformed_points(2, i);
//     }

//     // Publish the transformed point cloud.
//     pc_pub_->publish(output);
//   }

//   // --- Member variables ---
//   rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr odom_sub_;
//   rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_sub_;
//   rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pc_pub_;

//   // The latest odometry message received.
//   px4_msgs::msg::VehicleOdometry::SharedPtr latest_odom_;

//   // Fixed translation from camera to body.
//   Eigen::Vector3f t_camera_body_;
// };

// int main(int argc, char * argv[])
// {
//   rclcpp::init(argc, argv);
//   auto node = std::make_shared<PointCloudTransformer>();
//   rclcpp::spin(node);
//   rclcpp::shutdown();
//   return 0;
// }
/////////////////////////////////////////////////////////////////// owen code for sync //////////////////////////
// #include <memory>
// #include <vector>
// #include <mutex>
// #include "rclcpp/rclcpp.hpp"
// #include "sensor_msgs/msg/point_cloud2.hpp"
// #include "sensor_msgs/msg/point_field.hpp"
// #include "px4_msgs/msg/vehicle_odometry.hpp"
// #include "sensor_msgs/point_cloud2_iterator.hpp"
// #include "tf2/LinearMath/Quaternion.h"
// #include "tf2/LinearMath/Matrix3x3.h"
// #include "tf2/LinearMath/Vector3.h"

// class PointCloudTransformer : public rclcpp::Node
// {
// public:
//   PointCloudTransformer()
//   : Node("pointcloud_transformer")
//   {
//     // Subscribe to the high-frequency odometry topic.
//     odom_sub_ = this->create_subscription<px4_msgs::msg::VehicleOdometry>(
//       "/fmu/out/vehicle_odometry",
//       rclcpp::QoS(10).reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT),
//       std::bind(&PointCloudTransformer::odom_callback, this, std::placeholders::_1));

//     // Subscribe to the point cloud topic.
//     pc_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
//       "/depth_camera/points",
//       10,
//       std::bind(&PointCloudTransformer::pointcloud_callback, this, std::placeholders::_1));

//     // Publisher for the transformed point cloud.
//     pc_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/transformed_points", 10);

//     // Fixed transformation from camera to body.
//     R_camera_body_ = tf2::Matrix3x3(
//       0, 0, 1,
//       1, 0, 0,
//       0, 1, 0);
//     t_camera_body_ = tf2::Vector3(0.1, 0.0, -0.05);

//     // Initialize the latest odometry with zero timestamps.
//     latest_odom_.timestamp = 0;
//     latest_odom_.timestamp_sample = 0;
//   }

// private:
//   // Odometry callback: update the stored latest odometry.
//   void odom_callback(const px4_msgs::msg::VehicleOdometry::SharedPtr msg)
//   {
//     std::lock_guard<std::mutex> lock(odom_mutex_);
//     latest_odom_ = *msg;
//   }

//   // Point cloud callback: transform the point cloud using the latest odometry.
//   void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr pc_msg)
//   {
//     // Get a copy of the latest odometry data.
//     px4_msgs::msg::VehicleOdometry current_odom;
//     {
//       std::lock_guard<std::mutex> lock(odom_mutex_);
//       current_odom = latest_odom_;
//     }
//     if (current_odom.timestamp == 0) {
//       RCLCPP_WARN(this->get_logger(), "No odometry received yet; skipping point cloud transformation");
//       return;
//     }

//     // --- Compute body-to-map transformation ---
//     // PX4 VehicleOdometry message has a quaternion field 'q' (float32[4])
//     // The convention is assumed to be [w, x, y, z]. Adjust if needed.
//     double q_w = current_odom.q[0];
//     double q_x = current_odom.q[1];
//     double q_y = current_odom.q[2];
//     double q_z = current_odom.q[3];

//     // Construct a tf2 quaternion (note: tf2::Quaternion expects parameters in order (x, y, z, w)).
//     tf2::Quaternion q_tf(q_x, q_y, q_z, q_w);
//     tf2::Matrix3x3 R_body_map(q_tf);

//     // Compute translation from body to map.
//     // The VehicleOdometry message provides a position field (float32[3]).
//     // In this example we use a sign flip for y and z as in the original code.
//     tf2::Vector3 t_body_map(
//       current_odom.position[0],
//       -current_odom.position[1],
//       -current_odom.position[2]);

//     // --- Process the point cloud ---
//     std::vector<tf2::Vector3> transformed_points;
//     sensor_msgs::PointCloud2ConstIterator<float> iter_x(*pc_msg, "x");
//     sensor_msgs::PointCloud2ConstIterator<float> iter_y(*pc_msg, "y");
//     sensor_msgs::PointCloud2ConstIterator<float> iter_z(*pc_msg, "z");
//     for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
//       // Original point in the camera frame.
//       tf2::Vector3 p_camera(*iter_x, *iter_y, *iter_z);
//       // Transform from camera to body frame (fixed translation).
//       tf2::Vector3 p_body = p_camera + t_camera_body_;
//       // Transform from body to map frame (rotate then translate).
//       tf2::Vector3 p_map = R_body_map * p_body + t_body_map;
//       transformed_points.push_back(p_map);
//     }

//     // --- Create the output PointCloud2 message ---
//     sensor_msgs::msg::PointCloud2 output;
//     output.header.stamp = this->now();
//     output.header.frame_id = "ground_link";  // For example, the map frame.
//     output.height = 1;
//     output.width = transformed_points.size();
//     output.is_bigendian = false;
//     output.is_dense = true;
//     output.point_step = 12;  // 3 floats (x,y,z) × 4 bytes each.
//     output.row_step = output.point_step * output.width;

//     // Define the fields: x, y, and z.
//     output.fields.resize(3);
//     output.fields[0].name   = "x";
//     output.fields[0].offset = 0;
//     output.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
//     output.fields[0].count  = 1;

//     output.fields[1].name   = "y";
//     output.fields[1].offset = 4;
//     output.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
//     output.fields[1].count  = 1;

//     output.fields[2].name   = "z";
//     output.fields[2].offset = 8;
//     output.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
//     output.fields[2].count  = 1;

//     output.data.resize(output.row_step * output.height);
//     for (size_t i = 0; i < transformed_points.size(); ++i) {
//       float * ptr = reinterpret_cast<float*>(&output.data[i * output.point_step]);
//       ptr[0] = static_cast<float>(transformed_points[i].x());
//       ptr[1] = static_cast<float>(transformed_points[i].y());
//       ptr[2] = static_cast<float>(transformed_points[i].z());
//     }

//     pc_pub_->publish(output);
//   }

//   rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr odom_sub_;
//   rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_sub_;
//   rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pc_pub_;

//   // Latest odometry message and mutex for thread-safe access.
//   px4_msgs::msg::VehicleOdometry latest_odom_;
//   std::mutex odom_mutex_;

//   // Fixed transformation from camera to body.
//   tf2::Matrix3x3 R_camera_body_;
//   tf2::Vector3 t_camera_body_;
// };

// int main(int argc, char * argv[])
// {
//   rclcpp::init(argc, argv);
//   auto node = std::make_shared<PointCloudTransformer>();
//   rclcpp::spin(node);
//   rclcpp::shutdown();
//   return 0;
// }