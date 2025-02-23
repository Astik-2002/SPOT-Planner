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

class PointCloudTransformer : public rclcpp::Node
{
public:
  PointCloudTransformer()
  : Node("pointcloud_transformer")
  {
    // Set up the odometry subscriber with a BEST_EFFORT QoS and depth 1.
    auto qos_odometry = rclcpp::QoS(rclcpp::KeepLast(1));
    qos_odometry.reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT);
    odom_sub_ = this->create_subscription<px4_msgs::msg::VehicleOdometry>(
      "/fmu/out/vehicle_odometry", qos_odometry,
      std::bind(&PointCloudTransformer::odom_callback, this, std::placeholders::_1));

    // Set up the point cloud subscriber with a queue size of 10.
    pc_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/depth_camera/points", 10,
      std::bind(&PointCloudTransformer::pc_callback, this, std::placeholders::_1));

    // Publisher for the transformed point cloud.
    pc_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/transformed_points", 10);

    // Initialize the fixed transformation constants.
    // Although the Python code defines R_camera_body, note that it is not used in the transformation.
    R_camera_body_ = tf2::Matrix3x3(
      0, 0, 1,
      1, 0, 0,
      0, 1, 0);

    // Adjust this translation as necessary (units in meters).
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
    // The PX4 odometry message provides the quaternion as [w, x, y, z].
    double q_w = latest_odom_->q[0];
    double q_x = latest_odom_->q[1];
    double q_y = latest_odom_->q[2];
    double q_z = latest_odom_->q[3];

    // tf2::Quaternion expects (x, y, z, w)
    tf2::Quaternion q_tf(q_x, q_y, q_z, q_w);
    tf2::Matrix3x3 R_body_map(q_tf);

    // Compute translation from body to map.
    // Note the Python code negates the y and z components.
    tf2::Vector3 t_body_map(
      latest_odom_->position[0],
      -latest_odom_->position[1],
      -latest_odom_->position[2]);

    // --- Process the point cloud ---
    std::vector<tf2::Vector3> transformed_points;

    // Create iterators to go through the input point cloud.
    sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");

    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
      // Read the point from the camera.
      tf2::Vector3 p_camera(*iter_x, *iter_y, *iter_z);

      // Camera-to-body transform: add the fixed translation.
      tf2::Vector3 p_body = p_camera + t_camera_body_;

        tf2::Matrix3x3 Rz_90(
            0, -1,  0,
            1,  0,  0,
            0,  0,  1
        );
        // Body-to-map transform: rotate then translate.
        tf2::Vector3 p_map =  R_body_map * p_body + t_body_map;

      transformed_points.push_back(p_map);
    }

    // --- Create the output PointCloud2 message ---
    sensor_msgs::msg::PointCloud2 output;
    output.header.stamp = this->now();
    output.header.frame_id = "ground_link";  // Transformed into the map frame.
    output.height = 1;
    output.width = transformed_points.size();
    output.is_bigendian = false;
    output.is_dense = true;
    output.point_step = 12;  // 3 floats x 4 bytes each.
    output.row_step = output.point_step * output.width;

    // Define the fields: x, y, and z.
    output.fields.resize(3);
    output.fields[0].name   = "x";
    output.fields[0].offset = 0;
    output.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
    output.fields[0].count  = 1;

    output.fields[1].name   = "y";
    output.fields[1].offset = 4;
    output.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
    output.fields[1].count  = 1;

    output.fields[2].name   = "z";
    output.fields[2].offset = 8;
    output.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
    output.fields[2].count  = 1;

    // Allocate space for the point data.
    output.data.resize(output.row_step * output.height);

    // Copy the transformed points into the output data array.
    for (size_t i = 0; i < transformed_points.size(); ++i) {
      // Pointer to the start of the i-th point.
      float * ptr = reinterpret_cast<float*>(&output.data[i * output.point_step]);
      ptr[0] = static_cast<float>(transformed_points[i].x());
      ptr[1] = static_cast<float>(transformed_points[i].y());
      ptr[2] = static_cast<float>(transformed_points[i].z());
    }

    // Publish the transformed point cloud.
    pc_pub_->publish(output);
  }

  // --- Member variables ---
  rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pc_pub_;

  // Latest received odometry.
  px4_msgs::msg::VehicleOdometry::SharedPtr latest_odom_;

  // Fixed transformation from camera to body.
  tf2::Matrix3x3 R_camera_body_;  // (Not used in the transformation below.)
  tf2::Vector3 t_camera_body_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PointCloudTransformer>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
