#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <memory>
#include <mutex>

class PointCloudTransformer : public rclcpp::Node
{
public:
  PointCloudTransformer()
  : Node("pointcloud_transformer")
  {
    // Parameters
    this->declare_parameter("base_frame", "map");
    this->declare_parameter("child_frame", "base_link");
    this->declare_parameter("voxel_size", 0.4);
    
    base_frame_ = this->get_parameter("base_frame").as_string();
    child_frame_ = this->get_parameter("child_frame").as_string();
    voxel_size_ = this->get_parameter("voxel_size").as_double();

    // Subscribers
    odom_sub_ = this->create_subscription<px4_msgs::msg::VehicleOdometry>(
      "/fmu/out/vehicle_odometry", rclcpp::SensorDataQoS(),
      std::bind(&PointCloudTransformer::odometry_callback, this, std::placeholders::_1));
    
    pc_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/depth_camera/points", rclcpp::SensorDataQoS(),
      std::bind(&PointCloudTransformer::pointcloud_callback, this, std::placeholders::_1));

    // Publishers
    pc_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/transformed_pointcloud", 10);
    odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/odometry_enu", 10);
    goal_pub_ = this->create_publisher<nav_msgs::msg::Path>("/waypoints", 10);
    // TF Broadcaster
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    RCLCPP_INFO(this->get_logger(), "PointCloud Transformer node initialized");
  }

private:
  void odometry_callback(const px4_msgs::msg::VehicleOdometry::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(transform_mutex_);
    
    // Store timestamp for the transform
    rclcpp::Time stamp(msg->timestamp, RCL_ROS_TIME);

    // Create transform from NED to ENU (ROS convention)
    geometry_msgs::msg::TransformStamped transform;
    transform.header.stamp = stamp;
    transform.header.frame_id = base_frame_;
    transform.child_frame_id = child_frame_;

    // Position conversion from NED to ENU
    transform.transform.translation.x = msg->position[0];  // x
    transform.transform.translation.y = -msg->position[1];  // NED to ENU (y)
    transform.transform.translation.z = -msg->position[2];  // NED to ENU (z)

    // Orientation conversion from NED to ENU (q[0] is w in PX4)
    transform.transform.rotation.x = msg->q[1];
    transform.transform.rotation.y = -msg->q[2];  // NED to ENU
    transform.transform.rotation.z = -msg->q[3];  // NED to ENU
    transform.transform.rotation.w = msg->q[0];

    // Store the latest transform
    latest_transform_ = transform;
    
    // Broadcast the transform
    tf_broadcaster_->sendTransform(transform);

    // Also publish as Odometry message for visualization
    nav_msgs::msg::Odometry odom_msg;
    odom_msg.header = transform.header;
    odom_msg.child_frame_id = child_frame_;
    odom_msg.pose.pose.position.x = transform.transform.translation.x;
    odom_msg.pose.pose.position.y = transform.transform.translation.y;
    odom_msg.pose.pose.position.z = transform.transform.translation.z;
    odom_msg.pose.pose.orientation = transform.transform.rotation;
    
    // Convert velocity from NED to ENU
    odom_msg.twist.twist.linear.x = msg->velocity[0];
    odom_msg.twist.twist.linear.y = -msg->velocity[1];
    odom_msg.twist.twist.linear.z = -msg->velocity[2];
    
    odom_pub_->publish(odom_msg);

    nav_msgs::msg::Path goal_msg;
    geometry_msgs::msg::PoseStamped goal_pose;

    // Set the header information
    goal_msg.header.stamp = this->now();  // Use now() instead of get_clock()
    goal_msg.header.frame_id = base_frame_;

    // Set the pose header
    goal_pose.header = goal_msg.header;

    // Set the position (x, y, z)
    goal_pose.pose.position.x = 0.0;  // Your x coordinate
    goal_pose.pose.position.y = -50.0;  // Your y coordinate
    goal_pose.pose.position.z = 2.5;  // Your z coordinate

    // Set the orientation (identity orientation by default)
    goal_pose.pose.orientation.x = 0.0;
    goal_pose.pose.orientation.y = 0.0;
    goal_pose.pose.orientation.z = 0.0;
    goal_pose.pose.orientation.w = 1.0;
    goal_msg.poses.push_back(goal_pose);
    goal_pub_->publish(goal_msg);
    
  }

  void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(transform_mutex_);
    
    // Check if we have a valid transform
    if (latest_transform_.header.stamp.sec == 0 && 
        latest_transform_.header.stamp.nanosec == 0) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 1000,
        "No transform available yet, skipping point cloud");
      return;
    }

    // Convert ROS2 PointCloud2 to PCL
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *pcl_cloud);

    // Downsample using voxel grid filter
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl_cloud->size() > 0) {
      pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
      voxel_filter.setInputCloud(pcl_cloud);
      voxel_filter.setLeafSize(voxel_size_, voxel_size_, voxel_size_);
      voxel_filter.filter(*pcl_cloud_filtered);
    } else {
      pcl_cloud_filtered = pcl_cloud;
    }

    // Transform point cloud to map frame using the latest transform
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_transformed(new pcl::PointCloud<pcl::PointXYZ>);
    
    // Create transform from geometry_msgs::Transform
    tf2::Transform tf_transform;
    tf2::fromMsg(latest_transform_.transform, tf_transform);

    // Transform each point
    for (const auto& point : *pcl_cloud_filtered) {
      tf2::Vector3 point_in(point.x, point.y, point.z);
      tf2::Vector3 point_out = tf_transform * point_in;
      pcl_cloud_transformed->push_back(pcl::PointXYZ(
        static_cast<float>(point_out.x()),
        static_cast<float>(point_out.y()),
        static_cast<float>(point_out.z())));
    }

    // Convert back to ROS2 PointCloud2
    sensor_msgs::msg::PointCloud2 output_msg;
    pcl::toROSMsg(*pcl_cloud_transformed, output_msg);
    output_msg.header.stamp = msg->header.stamp;  // Keep original timestamp
    output_msg.header.frame_id = base_frame_;     // Publish in map frame

    pc_pub_->publish(output_msg);
  }

  // ROS2 components
  rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pc_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr goal_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  // Data storage
  geometry_msgs::msg::TransformStamped latest_transform_;
  std::mutex transform_mutex_;

  // Parameters
  std::string base_frame_;
  std::string child_frame_;
  double voxel_size_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PointCloudTransformer>());
  rclcpp::shutdown();
  return 0;
}