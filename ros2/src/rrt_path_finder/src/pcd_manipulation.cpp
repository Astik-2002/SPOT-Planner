#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <custom_interface_gym/msg/bounding_box_array.hpp>
#include <custom_interface_gym/msg/dynamic_bbox.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/message_filter.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

#include <algorithm>
#include <vector>
#include <utility>
#include <Eigen/Dense>

class DynamicObstacleFilterNode : public rclcpp::Node
{
public:
    DynamicObstacleFilterNode()
        : Node("dynamic_obstacle_filter_node"), tf_buffer_(get_clock()), tf_listener_(tf_buffer_)
    {
        using std::placeholders::_1;

        bbox_sub_ = this->create_subscription<custom_interface_gym::msg::BoundingBoxArray>(
            "/dynamic_obs_state", 10, std::bind(&DynamicObstacleFilterNode::bboxCallback, this, _1));

        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "noisy_pcd_gym_pybullet", rclcpp::SensorDataQoS(), std::bind(&DynamicObstacleFilterNode::rcvPointCloudCallBack, this, _1));

        dynamic_pcl_pub_0_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/dynamic_cloud_t0", 10);
        dynamic_pcl_pub_1_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/dynamic_cloud_t1", 10);
        dynamic_pcl_pub_2_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/dynamic_cloud_t2", 10);
        dynamic_pcl_pub_3_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/dynamic_cloud_t3", 10);
        dynamic_pcl_pub_4_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/dynamic_cloud_t4", 10);
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Subscription<custom_interface_gym::msg::BoundingBoxArray>::SharedPtr bbox_sub_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr dynamic_pcl_pub_0_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr dynamic_pcl_pub_1_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr dynamic_pcl_pub_2_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr dynamic_pcl_pub_3_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr dynamic_pcl_pub_4_;

    std::vector<Eigen::Matrix3d> dynamic_obs_array;
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> dynamic_points;

    pcl::PointCloud<pcl::PointXYZ> cloud_input;

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    const double delta_t = 1.0;  // Prediction step (seconds)

    void bboxCallback(const custom_interface_gym::msg::BoundingBoxArray::SharedPtr msg)
    {
        dynamic_obs_array.clear();
        for (const auto& bbox : msg->boxes)
        {
            Eigen::Matrix3d bbox_mat;
            bbox_mat(0, 0) = bbox.center_x;
            bbox_mat(0, 1) = bbox.center_y;
            bbox_mat(0, 2) = bbox.center_z;

            bbox_mat(1, 0) = bbox.velocity_x;
            bbox_mat(1, 1) = bbox.velocity_y;
            bbox_mat(1, 2) = bbox.velocity_z;

            bbox_mat(2, 0) = bbox.height;
            bbox_mat(2, 1) = bbox.length;
            bbox_mat(2, 2) = bbox.width;

            dynamic_obs_array.push_back(bbox_mat);
        }
    }

    void rcvPointCloudCallBack(const sensor_msgs::msg::PointCloud2::SharedPtr pointcloud_msg)
    {
        if (pointcloud_msg->data.empty()) return;

        sensor_msgs::msg::PointCloud2 cloud_transformed;
        try
        {
            tf_buffer_.transform(*pointcloud_msg, cloud_transformed, "ground_link", tf2::durationFromSec(0.1));
        }
        catch (tf2::TransformException& ex)
        {
            RCLCPP_WARN(this->get_logger(), "TF error: %s", ex.what());
            return;
        }

        cloud_input.clear();
        pcl::fromROSMsg(cloud_transformed, cloud_input);
        if (cloud_input.empty()) return;

        dynamic_points.clear();
        pcl::PointCloud<pcl::PointXYZ>::Ptr dynamic_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        dynamic_cloud->points.reserve(cloud_input.points.size() / 10);

        auto& points = cloud_input.points;
        auto new_end = std::remove_if(points.begin(), points.end(),
            [&](const pcl::PointXYZ& point) {
                for (const auto& bbox : dynamic_obs_array)
                {
                    double cx = bbox(0, 0), cy = bbox(0, 1), cz = bbox(0, 2);
                    double half_h = bbox(2, 0) / 2.0, half_l = bbox(2, 1) / 2.0, half_w = bbox(2, 2) / 2.0;
                    Eigen::Vector3d vel(bbox(1, 0), bbox(1, 1), bbox(1, 2));
                    if (point.x >= cx - half_l && point.x <= cx + half_l &&
                        point.y >= cy - half_w && point.y <= cy + half_w &&
                        point.z >= cz - half_h && point.z <= cz + half_h)
                    {
                        dynamic_cloud->points.push_back(point);
                        dynamic_points.emplace_back(Eigen::Vector3d(point.x, point.y, point.z), vel);
                        return true;
                    }
                }
                return false;
            });

        points.erase(new_end, points.end());

        publishFutureDynamicCloud(dynamic_cloud, 0, dynamic_pcl_pub_0_);
        for (int i = 1; i <= 4; ++i)
        {
            auto future_points = getObstaclePoints(i);
            publishColoredCloud(future_points, i);
        }
    }

    std::vector<Eigen::Vector3d> getObstaclePoints(int n)
    {
        std::vector<Eigen::Vector3d> future_pts;
        for (const auto& [pos, vel] : dynamic_points)
        {
            Eigen::Vector3d pred = pos + n * delta_t * vel;
            future_pts.push_back(pred);
        }
        return future_pts;
    }

    void publishFutureDynamicCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int n,
                                   rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub)
    {
        sensor_msgs::msg::PointCloud2 msg;
        pcl::toROSMsg(*cloud, msg);
        msg.header.frame_id = "ground_link";
        msg.header.stamp = this->get_clock()->now();
        pub->publish(msg);
    }

    void publishColoredCloud(const std::vector<Eigen::Vector3d>& points, int n)
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        for (const auto& pt : points)
        {
            pcl::PointXYZRGB p;
            p.x = pt.x();
            p.y = pt.y();
            p.z = pt.z();

            switch (n)
            {
            case 1: p.r = 255; p.g = 0;   p.b = 0;   break;
            case 2: p.r = 0;   p.g = 255; p.b = 0;   break;
            case 3: p.r = 0;   p.g = 0;   p.b = 255; break;
            case 4: p.r = 255; p.g = 255; p.b = 0;   break;
            }

            colored_cloud->points.push_back(p);
        }

        sensor_msgs::msg::PointCloud2 cloud_msg;
        pcl::toROSMsg(*colored_cloud, cloud_msg);
        cloud_msg.header.frame_id = "ground_link";
        cloud_msg.header.stamp = this->get_clock()->now();

        switch (n)
        {
        case 1: dynamic_pcl_pub_1_->publish(cloud_msg); break;
        case 2: dynamic_pcl_pub_2_->publish(cloud_msg); break;
        case 3: dynamic_pcl_pub_3_->publish(cloud_msg); break;
        case 4: dynamic_pcl_pub_4_->publish(cloud_msg); break;
        }
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DynamicObstacleFilterNode>());
    rclcpp::shutdown();
    return 0;
}
