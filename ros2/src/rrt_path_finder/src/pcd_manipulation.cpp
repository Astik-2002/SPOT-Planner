#include <iostream>
#include <fstream>
#include <Eigen/Eigen>
#include <math.h>
#include <random>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include "std_msgs/msg/float32_multi_array.hpp"
#include "custom_interface_gym/msg/traj_msg.hpp"
#include "custom_interface_gym/msg/des_trajectory.hpp"

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

class PointCloudManipulation : public rclcpp::Node
{
private:
    // TF2 buffer and listener
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr _map_sub;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr _inflated_pub;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr _noisy_pub;
    std::default_random_engine generator;

    int height = 48;
    int width = 64;
    float clip_distance = 8.0;

public:
    PointCloudManipulation() : Node("Point_Cloud_manipulation"),
                          tf_buffer_(this->get_clock()),
                          tf_listener_(tf_buffer_)
    {
        _map_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>("pcd_gym_pybullet", 1, std::bind(&PointCloudManipulation::rcvPointCloudCallback, this, std::placeholders::_1));
        _inflated_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("_inflated_pub", 1);
        _noisy_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("noisy_pcd_gym_pybullet", 1);
    }

    void rcvPointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr pointcloud_msg)
    {
        pcl::PointCloud<pcl::PointXYZ> cloud_input;
        pcl::fromROSMsg(*pointcloud_msg, cloud_input);
        pcl::PointCloud<pcl::PointXYZ> inflated_pcd;
        pcl::PointCloud<pcl::PointXYZ> noisy_pcd;

        for(auto pt: cloud_input.points)
        {
            double x = pt.x;
            double y = pt.y;
            double z = pt.z;
            
            double sigma_x = 0.0012 + 0.0019 * (z - 0.4) * (z - 0.4);// 0.001063 + 0.0007278 * x + 0.003949 * x * x;
            double sigma_y = 0.8*x/32;
            double sigma_z = sigma_y;
            std::normal_distribution<double> distribution_x(x, sigma_x);
            std::normal_distribution<double> distribution_y(y, sigma_y);
            std::normal_distribution<double> distribution_z(z, sigma_z);

            double noise_x = distribution_x(generator);
            double noise_y = distribution_y(generator);
            double noise_z = distribution_z(generator);
            pcl::PointXYZ noisy_p;
            noisy_p.x = noise_x;
            noisy_p.y = noise_y;
            noisy_p.z = noise_z;
            noisy_pcd.points.push_back(noisy_p);
        }
        sensor_msgs::msg::PointCloud2 noisy_cloud_msg;

        pcl::toROSMsg(noisy_pcd, noisy_cloud_msg);

        noisy_cloud_msg.header.frame_id = "base_link"; // Set appropriate frame ID
        noisy_cloud_msg.header.stamp = this->get_clock()->now();

        // _inflated_pub->publish(inflated_cloud_msg);
        _noisy_pub->publish(noisy_cloud_msg);
    }

};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PointCloudManipulation>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
