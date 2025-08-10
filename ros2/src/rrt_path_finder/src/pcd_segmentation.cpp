#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <Eigen/Dense>
#include "custom_interface_gym/msg/dynamic_bbox.hpp"
#include "custom_interface_gym/msg/bounding_box_array.hpp"
#include "custom_interface_gym/msg/image_array.hpp"
#include "custom_interface_gym/msg/pcd_array.hpp"
class DepthToPointCloud : public rclcpp::Node {
public:
    DepthToPointCloud() : Node("depth_to_pointcloud") {
        // Declare and get parameters
        this->declare_parameter("L", 0.039700);  // Near clipping plane (meters)
        this->declare_parameter("farVal", 1000.0);  // Far clipping plane (meters)
        this->declare_parameter("clip_distance", 30.0);  // Maximum valid distance (meters)

        L_ = this->get_parameter("L").as_double();
        farVal_ = this->get_parameter("farVal").as_double();
        clip_distance_ = this->get_parameter("clip_distance").as_double();

        // Define transformation matrices
        combined_rotation1 << 0, 0, 1, 0,
                              0, 1, 0, 0,
                             -1, 0, 0, 0,
                              0, 0, 0, 1;

        combined_rotation2 << 1, 0, 0, 0,
                              0, 0, 1, 0,
                              0, -1, 0, 0,
                              0, 0, 0, 1;

        opencv_to_ros <<  0, -1,  0, 0,  // X = -Y
                        0,  0, -1, 0,  // Y = -Z
                        1,  0,  0, 0,  // Z = X
                        0,  0,  0, 1;
        
        drone_transform_ = combined_rotation2 * combined_rotation1;
        // Compute final transformation matrix

        // Subscribe to depth image topic
        depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/depth_image", 10, std::bind(&DepthToPointCloud::depthCallback, this, std::placeholders::_1));
        
        bbox_sub = this->create_subscription<custom_interface_gym::msg::BoundingBoxArray>(
            "/dynamic_obs_state", 10, std::bind(&DepthToPointCloud::bboxCallback, this, std::placeholders::_1));
        
        image_array_sub = this->create_subscription<custom_interface_gym::msg::ImageArray>(
            "/depth_array", 10, std::bind(&DepthToPointCloud::arrayCallback, this, std::placeholders::_1));

        // Publish point cloud
        pcl_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("pcd_gym_pybullet", 10);

        pcl_pub1_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("noisy_pcd_gym_pybullet1", 10);
        pcl_pub2_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("noisy_pcd_gym_pybullet2", 10);
        pcl_pub3_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("noisy_pcd_gym_pybullet3", 10);
        pcl_pub4_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("noisy_pcd_gym_pybullet4", 10);
        pcl_array_pub_ = this->create_publisher<custom_interface_gym::msg::PcdArray>("pcd_array_gym_pybullet", 10);
        dynamic_pcl_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("dynamic_pcd_gym_pybullet", 10);

    }

private:
    void depthCallback(const sensor_msgs::msg::Image::SharedPtr msg) 
    {
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "CV Bridge conversion failed: %s", e.what());
            return;
        }

        int width = msg->width;
        int height = msg->height;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());

        pcl::PointCloud<pcl::PointXYZ>::Ptr dynamic_cloud(new pcl::PointCloud<pcl::PointXYZ>());

        double aspect = static_cast<double>(width) / height;
        double fov_y_rad = 90.0 * M_PI / 180.0;
        double fy = (height * 0.5) / std::tan(fov_y_rad * 0.5);

        // fx computed from horizontal FOV
        double fov_x_rad = 2.0 * std::atan(std::tan(fov_y_rad / 2.0) * aspect);
        double fx = (width * 0.5) / std::tan(fov_x_rad * 0.5);                    // horizontal
        double cx = (width) * 0.5;
        double cy = (height) * 0.5;

        // Process depth image into a 3D point cloud
        for (int v = 0; v < height; ++v) {
            for (int u = 0; u < width; ++u) 
            {
                float depth_image = cv_ptr->image.at<float>(v, u);
                if (depth_image <= 0.0) continue;  // Skip invalid depth
        
                // Convert normalized depth to real depth (m)
                
                double depth_m = (2.0 * L_ * farVal_) /
                    (farVal_ + L_ - (2.0 * depth_image - 1.0) * (farVal_ - L_)); 
                double Z = depth_m;
                double X = (u - cx) * Z / fx;
                double Y = (v - cy) * Z / fy;
        
                Eigen::Vector4d point_h(X, Y, Z, 1.0);
        
                Eigen::Vector4d transformed_pt = drone_transform_ * point_h;
                // Apply clipping (skip points beyond clip_distance_)
                if (Z < 0 || Z > clip_distance_) continue;
        
                // Add valid points to the point cloud
                pcl::PointXYZ point;
                point.x = static_cast<float>(transformed_pt(0));
                point.y = static_cast<float>(transformed_pt(1));
                point.z = static_cast<float>(transformed_pt(2));

                
                cloud->push_back(point);
            }
        }

        // Convert and publish point cloud
        sensor_msgs::msg::PointCloud2 cloud_msg;
        // RCLCPP_INFO(this->get_logger(), "Static cloud size: %zu, Dynamic cloud size: %zu", cloud->size(), dynamic_cloud->size());
        pcl::toROSMsg(*cloud, cloud_msg);
        cloud_msg.header.stamp = this->now();
        cloud_msg.header.frame_id = "base_link";  // Adjust frame if necessary
        pcl_pub_->publish(cloud_msg);
    }

    void bboxCallback(const custom_interface_gym::msg::BoundingBoxArray::SharedPtr msg)
    {
        dynamic_obs_array.clear();
        for(auto bbox : msg->boxes)
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

    void arrayCallback(const custom_interface_gym::msg::ImageArray::SharedPtr msg)
    {
        std::vector<sensor_msgs::msg::PointCloud2> cloud_vec;
        for(int i = 0; i < msg->images.size(); i++)  // Use actual array size
        {
            auto img_msg = msg->images[i];  // Access through images field
            cv_bridge::CvImagePtr cv_ptr;
            try {
                cv_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::TYPE_32FC1);
            } catch (cv_bridge::Exception& e) {
                RCLCPP_ERROR(this->get_logger(), "CV Bridge conversion failed: %s", e.what());
                continue;  // Skip this image but continue with others
            }

            int width = img_msg.width;  // Use image's own width
            int height = img_msg.height;  // Use image's own height
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());

            double aspect = static_cast<double>(width) / height;
            double fov_y_rad = 90.0 * M_PI / 180.0;
            double fy = (height * 0.5) / std::tan(fov_y_rad * 0.5);

            // fx computed from horizontal FOV
            double fov_x_rad = 2.0 * std::atan(std::tan(fov_y_rad / 2.0) * aspect);
            double fx = (width * 0.5) / std::tan(fov_x_rad * 0.5);
            double cx = width * 0.5;
            double cy = height * 0.5;

            // Process depth image into a 3D point cloud
            for (int v = 0; v < height; ++v) {
                for (int u = 0; u < width; ++u) 
                {
                    float depth_image = cv_ptr->image.at<float>(v, u);
                    if (depth_image <= 0.0) continue;  // Skip invalid depth
            
                    // Convert normalized depth to real depth (m)
                    double depth_m = (2.0 * L_ * farVal_) /
                        (farVal_ + L_ - (2.0 * depth_image - 1.0) * (farVal_ - L_)); 
                    double Z = depth_m;
                    double X = (u - cx) * Z / fx;
                    double Y = (v - cy) * Z / fy;
            
                    Eigen::Vector4d point_h(X, Y, Z, 1.0);
            
                    // Apply clipping (skip points beyond clip_distance_)
                    if (Z < 0 || Z > clip_distance_) continue;
            
                    // Add valid points to the point cloud
                    pcl::PointXYZ point;
                    point.x = static_cast<float>(point_h(0));
                    point.y = static_cast<float>(point_h(1));
                    point.z = static_cast<float>(point_h(2));
                    cloud->push_back(point);
                }
            }

            // Convert and publish point cloud with unique frame ID
            sensor_msgs::msg::PointCloud2 cloud_msg;
            pcl::toROSMsg(*cloud, cloud_msg);
            cloud_msg.header = img_msg.header;  // Preserve original timestamp
            cloud_msg.header.frame_id = "camera_link" + std::to_string(i+1);  // camera_link1, camera_link2, etc.
            cloud_vec.push_back(cloud_msg);

        }
        pcl_pub1_->publish(cloud_vec[0]);
        pcl_pub2_->publish(cloud_vec[1]);
        pcl_pub3_->publish(cloud_vec[2]);
        pcl_pub4_->publish(cloud_vec[3]);

        custom_interface_gym::msg::PcdArray cloud_array_msg;
        cloud_array_msg.pcds = cloud_vec;  // Assign the vector of pointclouds
        pcl_array_pub_->publish(cloud_array_msg);

    }
    // ROS 2 Parameters
    double L_, farVal_, clip_distance_;
    Eigen::Matrix4d drone_transform_;
    Eigen::Matrix4d opencv_to_ros;
    Eigen::Matrix4d combined_rotation1;
    Eigen::Matrix4d combined_rotation2;

    std::vector<Eigen::Matrix3d> dynamic_obs_array;
    // ROS 2 Subscriptions & Publishers
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
    rclcpp::Subscription<custom_interface_gym::msg::BoundingBoxArray>::SharedPtr bbox_sub;
    rclcpp::Subscription<custom_interface_gym::msg::ImageArray>::SharedPtr image_array_sub;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pcl_pub_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pcl_pub1_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pcl_pub2_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pcl_pub3_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pcl_pub4_;
    rclcpp::Publisher<custom_interface_gym::msg::PcdArray>::SharedPtr pcl_array_pub_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr dynamic_pcl_pub_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DepthToPointCloud>());
    rclcpp::shutdown();
    return 0;
}
