#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <Eigen/Dense>

class DepthToPointCloud : public rclcpp::Node {
public:
    DepthToPointCloud() : Node("depth_to_pointcloud") {
        // Declare and get parameters
        this->declare_parameter("L", 0.039700);  // Near clipping plane (meters)
        this->declare_parameter("farVal", 1000.0);  // Far clipping plane (meters)
        this->declare_parameter("clip_distance", 8.0);  // Maximum valid distance (meters)

        L_ = this->get_parameter("L").as_double();
        farVal_ = this->get_parameter("farVal").as_double();
        clip_distance_ = this->get_parameter("clip_distance").as_double();

        // Define transformation matrices
        Eigen::Matrix4d combined_rotation1;
        combined_rotation1 << 0, 0, 1, 0,
                              0, 1, 0, 0,
                             -1, 0, 0, 0,
                              0, 0, 0, 1;

        Eigen::Matrix4d combined_rotation2;
        combined_rotation2 << 1, 0, 0, 0,
                              0, 0, 1, 0,
                              0, -1, 0, 0,
                              0, 0, 0, 1;

        // Compute final transformation matrix
        drone_transform_ = combined_rotation2 * combined_rotation1;
        // drone_transform_ = drone_transform_.inverse().eval();  // Invert for extrinsic matrix

        // Subscribe to depth image topic
        depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/depth_noisy", 10, std::bind(&DepthToPointCloud::depthCallback, this, std::placeholders::_1));

        // Publish point cloud
        pcl_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("noisy_pcd_gym_pybullet", 10);
    }

private:
    void depthCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
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


        // Compute focal length dynamically based on FOV
        double aspect = static_cast<double>(width) / height;
        double fov = 90.0;
        double fx = width / (2.0 * std::tan(3.14/180*fov/2));
        double fy = height / (2.0 * std::tan(3.14/180*fov/2));
        double cx = width / 2.0;
        double cy = height / 2.0;

        // Process depth image into a 3D point cloud
        for (int v = 0; v < height; ++v) {
            for (int u = 0; u < width; ++u) {
                float depth_image = cv_ptr->image.at<float>(v, u);
                if (depth_image <= 0.0) continue;  // Skip invalid depth
        
                // Convert normalized depth to real depth (mm)
                double depth_mm = (2.0 * L_ * farVal_) / 
                                  (farVal_ + L_ - depth_image * (farVal_ - L_));
        
                double Z = depth_mm;
                double X = (u - cx) * Z / fx;
                double Y = (v - cy) * Z / fy;
        
                Eigen::Vector4d point_h(X, Y, Z, 1.0);
                Eigen::Vector4d transformed_point = drone_transform_ * point_h;
        
                double dist = transformed_point.head<3>().norm();  // Euclidean distance
        
                // Apply clipping (skip points beyond clip_distance_)
                if (Z < 0 || dist > clip_distance_) continue;
        
                // Add valid points to the point cloud
                pcl::PointXYZ point;
                point.x = static_cast<float>(transformed_point(0));
                point.y = static_cast<float>(transformed_point(1));
                point.z = static_cast<float>(transformed_point(2));
                cloud->push_back(point);
            }
        }

        // Convert and publish point cloud
        sensor_msgs::msg::PointCloud2 cloud_msg;
        pcl::toROSMsg(*cloud, cloud_msg);
        cloud_msg.header = msg->header;
        cloud_msg.header.frame_id = "base_link";  // Adjust frame if necessary
        pcl_pub_->publish(cloud_msg);
    }

    // ROS 2 Parameters
    double L_, farVal_, clip_distance_;
    Eigen::Matrix4d drone_transform_;

    // ROS 2 Subscriptions & Publishers
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pcl_pub_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DepthToPointCloud>());
    rclcpp::shutdown();
    return 0;
}
