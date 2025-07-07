#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <mavros_msgs/msg/state.hpp>
#include <mavros_msgs/msg/altitude.hpp>
#include <mavros_msgs/msg/extended_state.hpp>
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

    // MAVROS Subscribers
    local_pos_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      "/mavros/local_position/pose", rclcpp::SensorDataQoS(),
      std::bind(&PointCloudTransformer::local_position_callback, this, std::placeholders::_1));
    
    state_sub_ = this->create_subscription<mavros_msgs::msg::State>(
      "/mavros/state", rclcpp::SensorDataQoS(),
      [this](const mavros_msgs::msg::State::SharedPtr msg) {
        current_state_ = *msg;
      });
    
    // Point Cloud Subscriber
    pc_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/depth_camera/points", rclcpp::SensorDataQoS(),
      std::bind(&PointCloudTransformer::pointcloud_callback, this, std::placeholders::_1));

    // Publishers
    pc_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/transformed_pointcloud", 10);
    odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/odometry_enu", 10);
    goal_pub_ = this->create_publisher<nav_msgs::msg::Path>("/waypoints", 10);
    
    // TF Broadcaster
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    RCLCPP_INFO(this->get_logger(), "MAVROS PointCloud Transformer node initialized");
  }

private:
  void local_position_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {

    std::lock_guard<std::mutex> lock(transform_mutex_);
    
    // Publish map->odom transform (identity if not using SLAM)
    geometry_msgs::msg::TransformStamped map_to_odom;
    map_to_odom.header.stamp = msg->header.stamp;
    map_to_odom.header.frame_id = "map";
    map_to_odom.child_frame_id = "odom";
    map_to_odom.transform.rotation.w = 1.0; // Identity transform
    tf_broadcaster_->sendTransform(map_to_odom);

    // Create odom->base_link transform from MAVROS local position
    geometry_msgs::msg::TransformStamped odom_to_base;
    odom_to_base.header = msg->header;
    odom_to_base.header.frame_id = "odom";  // Changed from "map"
    odom_to_base.child_frame_id = child_frame_; // Typically "base_link"
    
    // Position (ENU)
    odom_to_base.transform.translation.x = msg->pose.position.x;
    odom_to_base.transform.translation.y = msg->pose.position.y;
    odom_to_base.transform.translation.z = msg->pose.position.z;
    
    // Orientation
    odom_to_base.transform.rotation = msg->pose.orientation;
    
    // Store and broadcast
    latest_transform_ = odom_to_base;
    tf_broadcaster_->sendTransform(odom_to_base);

    // Publish as Odometry message
    nav_msgs::msg::Odometry odom_msg;
    odom_msg.header = odom_to_base.header;
    odom_msg.child_frame_id = child_frame_;
    odom_msg.pose.pose.position.x = odom_to_base.transform.translation.x;
    odom_msg.pose.pose.position.y = odom_to_base.transform.translation.y;
    odom_msg.pose.pose.position.z = odom_to_base.transform.translation.z;
    odom_msg.pose.pose.orientation = odom_to_base.transform.rotation;
    
    // Note: Velocity would need to come from /mavros/local_position/velocity_local
    odom_pub_->publish(odom_msg);

    // Publish goal waypoints
    nav_msgs::msg::Path goal_msg;
    geometry_msgs::msg::PoseStamped goal_pose;

    goal_msg.header.stamp = this->now();
    goal_msg.header.frame_id = base_frame_;

    goal_pose.header = goal_msg.header;
    goal_pose.pose.position.x = 50.0;
    goal_pose.pose.position.y = 0.0;
    goal_pose.pose.position.z = 2.5;
    goal_pose.pose.orientation.w = 1.0;  // Neutral orientation
    
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
    // pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    // if (pcl_cloud->size() > 0) {
    //   pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    //   voxel_filter.setInputCloud(pcl_cloud);
    //   voxel_filter.setLeafSize(voxel_size_, voxel_size_, voxel_size_);
    //   voxel_filter.filter(*pcl_cloud_filtered);
    // } else {
    //   pcl_cloud_filtered = pcl_cloud;
    // }

    // Transform point cloud to map frame using the latest transform
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_transformed(new pcl::PointCloud<pcl::PointXYZ>);
    
    // Create transform from geometry_msgs::Transform
    tf2::Transform tf_transform;
    tf2::fromMsg(latest_transform_.transform, tf_transform);

    // Transform each point
    for (const auto& point : *pcl_cloud) {
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

  // Member variables
  std::string base_frame_;
  std::string child_frame_;
  double voxel_size_;
  geometry_msgs::msg::TransformStamped latest_transform_;
  mavros_msgs::msg::State current_state_;
  std::mutex transform_mutex_;
  
  // Subscribers
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr local_pos_sub_;
  rclcpp::Subscription<mavros_msgs::msg::State>::SharedPtr state_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_sub_;
  
  // Publishers
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pc_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr goal_pub_;
  
  // TF Broadcaster
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PointCloudTransformer>());
  rclcpp::shutdown();
  return 0;
}