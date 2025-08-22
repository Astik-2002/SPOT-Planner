#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <custom_interface_gym/msg/bounding_box_array.hpp>
#include <custom_interface_gym/msg/dynamic_bbox.hpp>
#include <custom_interface_gym/msg/dynamic_point.hpp>
#include <custom_interface_gym/msg/dynamic_point_cloud.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

#include <algorithm>
#include <vector>
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
            "pcd_gym_pybullet", rclcpp::SensorDataQoS(), std::bind(&DynamicObstacleFilterNode::rcvPointCloudCallBack, this, _1));
        
        static_pcl_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/static_cloud", 10);
        dynamic_structured_pub_ = this->create_publisher<custom_interface_gym::msg::DynamicPointCloud>("/dynamic_cloud_structured", 10);

        // Create N+1 publishers (t0 to tN)
        for (int i = 0; i <= n_preds_; ++i)
        {
            std::string topic = "/dynamic_cloud_t" + std::to_string(i);
            dynamic_pcl_pubs_.push_back(this->create_publisher<sensor_msgs::msg::PointCloud2>(topic, 10));
        }
    }

private:
    const double delta_t_ = 1.0;    // Time step in seconds
    const int n_preds_ = 10;        // Number of future prediction steps

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Subscription<custom_interface_gym::msg::BoundingBoxArray>::SharedPtr bbox_sub_;

    std::vector<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr> dynamic_pcl_pubs_;

    // Add publishers
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr static_pcl_pub_;
    rclcpp::Publisher<custom_interface_gym::msg::DynamicPointCloud>::SharedPtr dynamic_structured_pub_;

    std::vector<Eigen::Matrix3d> dynamic_obs_array_;
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> dynamic_points_;

    pcl::PointCloud<pcl::PointXYZ> cloud_input_;

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    void bboxCallback(const custom_interface_gym::msg::BoundingBoxArray::SharedPtr msg)
    {
        dynamic_obs_array_.clear();
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

            dynamic_obs_array_.push_back(bbox_mat);
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

        cloud_input_.clear();
        pcl::fromROSMsg(cloud_transformed, cloud_input_);
        if (cloud_input_.empty()) return;

        dynamic_points_.clear();
        pcl::PointCloud<pcl::PointXYZ>::Ptr dynamic_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        dynamic_cloud->points.reserve(cloud_input_.points.size() / 10);

        auto& points = cloud_input_.points;
        auto new_end = std::remove_if(points.begin(), points.end(),
            [&](const pcl::PointXYZ& point) {
                for (const auto& bbox : dynamic_obs_array_)
                {
                    double cx = bbox(0, 0), cy = bbox(0, 1), cz = bbox(0, 2);
                    double half_h = bbox(2, 0) / 2.0, half_l = bbox(2, 1) / 2.0, half_w = bbox(2, 2) / 2.0;
                    Eigen::Vector3d vel(bbox(1, 0), bbox(1, 1), bbox(1, 2));
                    if (point.x >= cx - half_l && point.x <= cx + half_l &&
                        point.y >= cy - half_w && point.y <= cy + half_w &&
                        point.z >= cz - half_h && point.z <= cz + half_h)
                    {
                        dynamic_cloud->points.push_back(point);
                        dynamic_points_.emplace_back(Eigen::Vector3d(point.x, point.y, point.z), vel);
                        return true;
                    }
                }
                return false;
            });

        points.erase(new_end, points.end());
        points.shrink_to_fit();

        cloud_input_.width = cloud_input_.points.size();
        cloud_input_.height = 1;  // unorganized point cloud
        cloud_input_.is_dense = false;

        sensor_msgs::msg::PointCloud2 static_msg;
        pcl::toROSMsg(cloud_input_, static_msg); // cloud_input_ now has only static points
        static_msg.header.frame_id = "ground_link";
        static_msg.header.stamp = this->get_clock()->now();
        static_pcl_pub_->publish(static_msg);

        // Publish dynamic structured cloud
        custom_interface_gym::msg::DynamicPointCloud dyn_msg;
        dyn_msg.header.frame_id = "ground_link";
        dyn_msg.header.stamp = this->get_clock()->now();

        for (const auto& [pos, vel] : dynamic_points_)
        {
            custom_interface_gym::msg::DynamicPoint dp;
            dp.position.x = pos.x();
            dp.position.y = pos.y();
            dp.position.z = pos.z();

            dp.velocity.x = vel.x();
            dp.velocity.y = vel.y();
            dp.velocity.z = vel.z();

            dyn_msg.points.push_back(dp);
        }
        dynamic_structured_pub_->publish(dyn_msg);
        // Publish current dynamic cloud
        // publishFutureDynamicCloud(dynamic_cloud, 0);

        // // Publish N future predictions
        // for (int i = 1; i <= n_preds_; ++i)
        // {
        //     auto future_points = getPredictedPoints(i);
        //     publishColoredCloud(future_points, i);
        // }
    }

    std::vector<Eigen::Vector3d> getPredictedPoints(int step)
    {
        std::vector<Eigen::Vector3d> future_pts;
        for (const auto& [pos, vel] : dynamic_points_)
        {
            Eigen::Vector3d pred = pos + step * delta_t_ * vel;
            future_pts.push_back(pred);
        }
        return future_pts;
    }

    void publishFutureDynamicCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int idx)
    {
        if (idx < 0 || idx > n_preds_) return;

        sensor_msgs::msg::PointCloud2 msg;
        pcl::toROSMsg(*cloud, msg);
        msg.header.frame_id = "ground_link";
        msg.header.stamp = this->get_clock()->now();
        dynamic_pcl_pubs_[idx]->publish(msg);
    }

    void publishColoredCloud(const std::vector<Eigen::Vector3d>& points, int idx)
    {
        if (idx < 1 || idx > n_preds_) return;

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        for (const auto& pt : points)
        {
            pcl::PointXYZRGB p;
            p.x = pt.x();
            p.y = pt.y();
            p.z = pt.z();

            // Unique color per prediction step
            p.r = (idx * 53) % 256;
            p.g = (idx * 97) % 256;
            p.b = (idx * 193) % 256;

            colored_cloud->points.push_back(p);
        }

        sensor_msgs::msg::PointCloud2 msg;
        pcl::toROSMsg(*colored_cloud, msg);
        msg.header.frame_id = "ground_link";
        msg.header.stamp = this->get_clock()->now();
        dynamic_pcl_pubs_[idx]->publish(msg);
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DynamicObstacleFilterNode>());
    rclcpp::shutdown();
    return 0;
}
