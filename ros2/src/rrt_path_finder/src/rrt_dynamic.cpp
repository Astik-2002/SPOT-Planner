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
#include "rrt_path_finder/corridor_finder.h"
// #include <px4_msgs/msg/vehicle_odometry.hpp>
// #include "px4_msgs/msg/vehicle_local_position.hpp"
#include "../utils/header/type_utils.hpp"
#include "../utils/header/eigen_alias.hpp"

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include "geometry_msgs/msg/pose_stamped.hpp"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <rclcpp/clock.hpp>
#include "rrt_path_finder/firi.hpp"
#include "rrt_path_finder/ciri.h"
#include "rrt_path_finder/ciri_ellip.h"
#include "rrt_path_finder/ciri_sphere.h"
#include "rrt_path_finder/gcopter.hpp"
#include "rrt_path_finder/gcopter_fixed.hpp"
#include "rrt_path_finder/trajectory.hpp"
#include "rrt_path_finder/geo_utils.hpp"
#include "rrt_path_finder/quickhull.hpp"
#include "rrt_path_finder/voxel_map.hpp"
#include "rrt_path_finder/corridor_finder_dynamic.h"
#include "rrt_path_finder/datatype_dynamic.h"
#include "rrt_path_finder/yaw_optimization.hpp"
#include "rrt_path_finder/non_uniform_bspline.hpp"
// #include "rrt_path_finder/custom_hash.hpp"
#include "custom_interface_gym/msg/dynamic_bbox.hpp"
#include "custom_interface_gym/msg/bounding_box_array.hpp"
#include "custom_interface_gym/msg/pcd_array.hpp"
#include "custom_interface_gym/msg/dynamic_point.hpp"
#include "custom_interface_gym/msg/dynamic_point_cloud.hpp"
#include "custom_interface_gym/msg/server_state.hpp"

using namespace std;
using namespace Eigen;
using namespace pcl;

enum PlannerState {
    INITIAL,
    INCREMENTAL,
    BACKUP
};

class PointCloudPlanner : public rclcpp::Node
{
public:
    PointCloudPlanner() : Node("point_cloud_planner"),
                          tf_buffer_(this->get_clock()),
                          tf_listener_(tf_buffer_),
                          gen(rd()),
                          dynamic_cloud(new pcl::PointCloud<pcl::PointXYZ>())
    {
        // Parameters
        this->declare_parameter("safety_margin", 1.0);
        this->declare_parameter("uav_radius", 0.5);
        this->declare_parameter("search_margin", 0.3);
        this->declare_parameter("max_radius", 5.0);
        this->declare_parameter("refine_portion", 0.80);
        this->declare_parameter("sample_portion", 0.25);
        this->declare_parameter("goal_portion", 0.05);
        this->declare_parameter("path_find_limit", 5.0);
        this->declare_parameter("max_samples", 10000);
        this->declare_parameter("stop_horizon", 0.5);
        this->declare_parameter("commit_time", 0.5);

        this->declare_parameter("x_l", -5.0);
        this->declare_parameter("x_h", 70.0);
        this->declare_parameter("y_l", -7.0);
        this->declare_parameter("y_h", 7.0);
        // this->declare_parameter("z_l", 1.0);
        this->declare_parameter("z_l2", 0.8);
        this->declare_parameter("z_l", 0.8);

        // this->declare_parameter("z_h", 1.0);
        this->declare_parameter("z_h2", 3.0);
        this->declare_parameter("z_h", 3.0);

        this->declare_parameter("target_x", 0.0);   
        this->declare_parameter("target_y", 0.0);
        this->declare_parameter("target_z", 0.0);
        this->declare_parameter("goal_input", true);
        this->declare_parameter("is_limit_vel", true);
        this->declare_parameter("is_limit_acc", true);
        this->declare_parameter("is_print", true);

        // Read parameters
        _safety_margin = this->get_parameter("safety_margin").as_double();
        _uav_radius = this->get_parameter("uav_radius").as_double();
        _search_margin = this->get_parameter("search_margin").as_double();
        _max_radius = this->get_parameter("max_radius").as_double();
        _refine_portion = this->get_parameter("refine_portion").as_double();
        _sample_portion = this->get_parameter("sample_portion").as_double();
        _goal_portion = this->get_parameter("goal_portion").as_double();
        _path_find_limit = this->get_parameter("path_find_limit").as_double();
        _max_samples = this->get_parameter("max_samples").as_int();
        _stop_time = this->get_parameter("stop_horizon").as_double();
        commit_time = this->get_parameter("commit_time").as_double();
        _x_l = this->get_parameter("x_l").as_double();
        _x_h = this->get_parameter("x_h").as_double();
        _y_l = this->get_parameter("y_l").as_double();
        _y_h = this->get_parameter("y_h").as_double();
        _z_l = this->get_parameter("z_l").as_double();
        _z_h = this->get_parameter("z_h").as_double();
        _z_h2 = this->get_parameter("z_h2").as_double();

        _z_l2 = this->get_parameter("z_l2").as_double();
        Eigen::Vector3i xyz((_x_h-_x_l)/voxelWidth, (_y_h-_y_l)/voxelWidth, (_z_h2-_z_l2)/voxelWidth);
        Eigen::Vector3d offset(_x_l, _y_l, _z_l2);
        V_map = voxel_map::VoxelMap(xyz, offset, voxelWidth);
        // Set parameters for RRT planner once
        setRRTPlannerParams();

        // Publishers
        // _vis_corridor_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>("flight_corridor", 1);
        _vis_rrt_tree_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>("_vis_rrt_tree", 1);
        _vis_corridor_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>("_viscorridor", 1);
        _vis_rrt_path_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>("_vis_rrt_path",1);
        _vis_map_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("_vis_pcd", 1);
        _vis_mesh_pub = this->create_publisher<visualization_msgs::msg::Marker>("_vis_mesh", 10);
        _vis_edge_pub = this->create_publisher<visualization_msgs::msg::Marker>("_vis_edge", 10);
        _vis_commit_target = this->create_publisher<visualization_msgs::msg::Marker>("_vis_commit_target", 10);
        _vis_collision_point = this->create_publisher<visualization_msgs::msg::Marker>("_vis_collision_point", 10);

        _vis_ellipsoid = this->create_publisher<visualization_msgs::msg::MarkerArray>("_vis_ellipsoid_marker", 10);

        _vis_trajectory_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("_vis_trajectory", 10);

        // Add the RRT waypoints publisher
        _rrt_waypoints_pub = this->create_publisher<nav_msgs::msg::Path>("rrt_waypoints", 1);
        _rrt_des_traj_pub = this->create_publisher<custom_interface_gym::msg::DesTrajectory>("des_trajectory",10);
        _corridor_endpoint_pub = this->create_publisher<geometry_msgs::msg::PoseStamped>("corridor_endpoint", 1); // For yaw control
        pcl_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("static_pointcloud", 10);
        dynamic_pcl_pub_1_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("dynamic_pcd_gym_pybullet_1", 10);
        dynamic_pcl_pub_2_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("dynamic_pcd_gym_pybullet_2", 10);
        dynamic_pcl_pub_3_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("dynamic_pcd_gym_pybullet_3", 10);
        dynamic_pcl_pub_4_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("dynamic_pcd_gym_pybullet_4", 10);
        dynamic_pcl_pub_0_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("dynamic_pcd_gym_pybullet_0", 10);

        // Subscribers
        // _obs_sub = this->create_subscription<std_msgs::msg::Float32MultiArray>(
        // "obs", 1, std::bind(&PointCloudPlanner::rcvObsCallback, this, std::placeholders::_1));

        _dest_pts_sub = this->create_subscription<nav_msgs::msg::Path>(
            "waypoints", 1, std::bind(&PointCloudPlanner::rcvWaypointsCallBack, this, std::placeholders::_1));
        _map_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "pcd_gym_pybullet", 1, std::bind(&PointCloudPlanner::rcvPointCloudCallBack, this, std::placeholders::_1));

        _odometry_sub = this->create_subscription<nav_msgs::msg::Odometry>(
            "odom", 10, std::bind(&PointCloudPlanner::rcvOdomCallback, this, std::placeholders::_1));
        _bbox_sub = this->create_subscription<custom_interface_gym::msg::BoundingBoxArray>(
            "/dynamic_obs_state", 10, std::bind(&PointCloudPlanner::bboxCallback, this, std::placeholders::_1));
        _dynamic_structured_sub = this->create_subscription<custom_interface_gym::msg::DynamicPointCloud>(
            "/dynamic_cloud_structured", 10, std::bind(&PointCloudPlanner::rcvDynamicPointsCallback, this, std::placeholders::_1));
        _server_state_sub = this->create_subscription<custom_interface_gym::msg::ServerState>(
            "server_state", rclcpp::QoS(10).reliable(),std::bind(&PointCloudPlanner::rcvServerStateCallback, this, std::placeholders::_1));


        // Create N+1 publishers (t0 to tN)
        for (int i = 0; i <= n_preds_; ++i)
        {
            std::string topic = "/dynamic_cloud_t" + std::to_string(i);
            dynamic_pcl_pubs_.push_back(this->create_publisher<sensor_msgs::msg::PointCloud2>(topic, 10));
        }

        // Timer for planning
        _planning_timer = this->create_wall_timer(
            std::chrono::duration<double>(0.2),
            std::bind(&PointCloudPlanner::planningCallBack, this));
    };

private:
    // Visualization Publishers
    // rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr _vis_corridor_pub;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr _vis_rrt_tree_pub;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr _vis_rrt_path_pub;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr _vis_corridor_pub;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr _vis_commit_target;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr _vis_collision_point;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr _vis_ellipsoid;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr _vis_map_pub;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr _vis_mesh_pub;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr _vis_edge_pub;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr _vis_trajectory_pub;

    // RRT waypoints publisher
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr _rrt_waypoints_pub;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr _corridor_endpoint_pub;
    rclcpp::Publisher<custom_interface_gym::msg::DesTrajectory>::SharedPtr _rrt_des_traj_pub;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pcl_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr dynamic_pcl_pub_1_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr dynamic_pcl_pub_2_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr dynamic_pcl_pub_3_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr dynamic_pcl_pub_4_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr dynamic_pcl_pub_0_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr _dest_pts_sub;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr _map_sub;    
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr _odometry_sub;
    rclcpp::Subscription<custom_interface_gym::msg::BoundingBoxArray>:: SharedPtr _bbox_sub;
    rclcpp::Subscription<custom_interface_gym::msg::DynamicPointCloud>::SharedPtr _dynamic_structured_sub;
    rclcpp::Subscription<custom_interface_gym::msg::ServerState>::SharedPtr _server_state_sub;
    // Timer
    rclcpp::TimerBase::SharedPtr _planning_timer;

    // TF2 buffer and listener
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    // Path Planning Parameters
    double _planning_rate, _safety_margin, commit_time, _uav_radius, _search_margin, _max_radius, _replan_distance;
    double _refine_portion, _sample_portion, _goal_portion, _path_find_limit, _stop_time, _time_commit;
    double _x_l, _x_h, _y_l, _y_h, _z_l, _z_h, _z_l2, _z_h2;  // For random map simulation: map boundary
    int n_preds_ = 10;        // Number of future prediction steps
    std::vector<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr> dynamic_pcl_pubs_;

    std::vector<Eigen::MatrixX4d> hpolys; // Matrix to store hyperplanes
    std::vector<Eigen::MatrixX4d> bkup_hpolys; // Matrix to store hyperplanes

    std::vector<Eigen::Vector3d> pcd_points; // Datastructure to hold pointcloud points in a vector
    std::vector<Eigen::Vector3d> corridor_points; // vector for storing points for which corridor is contructed
    std::vector<Eigen::Vector4d> corridor_points4d;
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> dynamic_points;

    tf2::Quaternion enu_q_global;
    
    rclcpp::Time trajstamp, bkup_trajstamp;
    rclcpp::Time odom_time;
    rclcpp::Time pointcloud_receive_time, init_planning_time;
    std::chrono::time_point<std::chrono::steady_clock> obsvstamp = std::chrono::steady_clock::now();
    int quadratureRes = 16;
    float smoothingEps = 0.01;
    float relcostto1 = 0.00001;
    int _max_samples;
    double _commit_distance = 6.0;
    double current_yaw = 0;
    double max_vel = 1.0;
    float threshold = 1.0;
    int trajectory_id = 0;
    int order = 5;
    int pcd_idx = 0;
    double convexCoverRange = 1.0;
    float convexDecompTime = 0.05;
    float traj_gen_time = 0.1;
    float bkup_convexDecompTime = 0.05;
    float bkup_traj_gen_time = 0.1;
    // RRT Path Planner
    safeRegionRrtStarDynamic _rrtPathPlanner;
    // safeRegionRrtStarDynamicEllip _rrtPathPlanner;
    gcopter::GCOPTER_PolytopeSFC _gCopter;
    gcopter_fixed::GCOPTER_PolytopeSFC_FixedTime _gCopter_fixed;
    pcl::PointCloud<pcl::PointXYZ>::Ptr dynamic_cloud;
    pcl::search::KdTree<pcl::PointXYZ> static_kdtree;
    pcl::search::KdTree<pcl::PointXYZ> dynamic_kdtree;

    Trajectory<5> _traj;
    super_planner::CIRI_e ciri_e;
    super_planner::CIRI_s ciri_s;
    super_planner::CIRI ciri;
    PlannerState current_state = INITIAL;
    yawOptimizer yaw_opt;
    Eigen::Vector3d pcd_origin;
    voxel_map::VoxelMap V_map;
    int max_iter=100000;
    float voxelWidth = 0.35;
    float dilateRadius = 1.0;
    float leafsize = 0.4;
    // Variables for target position, trajectory, odometry, etc.
    Eigen::Vector3d _start_pos, _end_pos{50.0, 0.0, 1.0}, bkup_goal{50.0, 0.0, 1.0} , _start_vel{0.0, 0.0, 0.0}, _last_vel{0.0, 0.0, 0.0}, _start_acc;
    Eigen::Vector4d _commit_target{15.0, 0.0, 0.0, 0.0};
    Eigen::Vector3d _corridor_end_pos;
    std::vector<geometry_utils::Ellipsoid> tangent_obs;
    super_utils::Mat3f rotation_matrix = super_utils::Mat3f::Identity();
    // uav physical params
    float mass = 0.027000;
    float horizontal_drag_coeff = 0.000001;
    float vertical_drag_coeff = 0.000001;
    float t2w = 5.0;
    double delta_t = 1.0;
    Eigen::MatrixXd _path;
    Eigen::VectorXd _radius;
    std::vector<Eigen::Vector4d> _path_vector;
    std::vector<Eigen::Matrix3d> dynamic_obs_array;
    std::vector<double> _radius_vector;
    std::vector<double> time_vector_poly;
    nav_msgs::msg::Odometry _odom;
    bool _is_traj_exist = false;
    bool _is_bkup_traj_exist = false;
    bool _is_target_arrive = false;
    bool _is_target_receive = false;
    bool _is_has_map = false;
    bool _is_complete = false;
    bool _is_yaw_enabled = true;
    bool uncertanity_compensation = true;
    bool dynamic_bkup = true;
    Eigen::Vector3d bkup_start_pos;
    pcl::PointCloud<pcl::PointXYZ> cloud_input;
    double PCDstart_time = 0.0;
    // Get current ROS clock time (now)
    rclcpp::Time now_ros;
    std::vector<int>     pointIdxRadiusSearch;
    std::vector<float>   pointRadiusSquaredDistance;
    bool force_test_backup = false, disable_bkup = true;  // Set to false after testing
    bool near_dynamic = false;
    bool _server_active = false;
    double t_server = -1.0;
    std::random_device rd;
    std::mt19937 gen;
    double weight_t = 0.3;
    std::unordered_map<Eigen::Vector3d, Eigen::Vector3d, Vec3dHash, Vec3dEqual> dynamic_points_hash{0, Vec3dHash(1e-6), Vec3dEqual(1e-6)};
    int num_bkup = 0.0;
    double time_initial = 0.0;
    double time_incremental = 0.0;
    double time_bkup = 0.0;
    float random_between(float lower, float upper) 
    {
        std::uniform_real_distribution<float> dis(lower, upper);
        return dis(gen);
    }

    void rcvServerStateCallback(const custom_interface_gym::msg::ServerState::SharedPtr msg)
    {
        if (msg->state == "ACTIVE")
        {
            _server_active = true;
            t_server = msg->elapsed_t;
        }
        else if(msg->state == "IDLE")
        {
            _server_active = false;
            t_server = -1.0;
        }
    }

    void rcvWaypointsCallBack(const nav_msgs::msg::Path::SharedPtr wp_msg)
    {
        if(_is_target_receive) return;
        if (wp_msg->poses.empty() || wp_msg->poses[0].pose.position.z < 0.0)
            return;

        _end_pos(0) = wp_msg->poses[0].pose.position.x;
        _end_pos(1) = wp_msg->poses[0].pose.position.y;
        _end_pos(2) = wp_msg->poses[0].pose.position.z;

        _is_target_receive = true;
        _is_target_arrive = false;
        _is_traj_exist = false;
        //RCLCPP_WARN(this->get_logger(), "Waypoint received");
    }

    // Initializing rrt parameters
    void setRRTPlannerParams()
    {
        // _rrtPathPlanner.setParam(_safety_margin, _search_margin, _max_radius, _commit_distance, 90, 90, uncertanity_compensation, _uav_radius);
        _rrtPathPlanner.setParam(_safety_margin, _search_margin, delta_t, _commit_distance, 90, 90, uncertanity_compensation);

        _rrtPathPlanner.reset();
        now_ros = rclcpp::Clock().now();

    }

    void rcvOdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        _odom = *msg;
        _start_pos[0] = _odom.pose.pose.position.x;
        _start_pos[1] = _odom.pose.pose.position.y;
        _start_pos[2] = _odom.pose.pose.position.z;

        _start_vel[0] = _odom.twist.twist.linear.x;
        _start_vel[1] = _odom.twist.twist.linear.y;
        _start_vel[2] = _odom.twist.twist.linear.z;

        tf2::Quaternion q(
        _odom.pose.pose.orientation.x,
        _odom.pose.pose.orientation.y,
        _odom.pose.pose.orientation.z,
        _odom.pose.pose.orientation.w
        );

        // Convert to Euler angles
        double roll, pitch, yaw;
        tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
        checkSafeTrajectory();
        // Store the yaw angle
        current_yaw = yaw; // Yaw in radians
        auto current_time = rclcpp::Time(_odom.header.stamp.sec, _odom.header.stamp.nanosec);
        auto del_t = (current_time - odom_time).seconds();
        _start_acc = (_start_vel - _last_vel)/(del_t);
        odom_time = rclcpp::Time(_odom.header.stamp.sec, _odom.header.stamp.nanosec);
        double dis_commit = _rrtPathPlanner.getDis(_start_pos, _commit_target.head<3>());
        // std::cout<<"distance between commit target and uav: "<<dis_commit<<std::endl;

        if(dis_commit < 1.0)
        {
            _is_target_arrive = true;
        }
        else
        {
            _is_target_arrive = false;
        }
        if(_is_target_receive && (_end_pos - _start_pos).norm() < 2.0)
        {
            std::cout<<"[debug]: "<<(_end_pos - _start_pos).norm()<<std::endl;
            _is_complete = true;
        }
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

    void rcvDynamicPointsCallback(const custom_interface_gym::msg::DynamicPointCloud::SharedPtr dynamic_pcd_msg)
    {
        pointcloud_receive_time = rclcpp::Clock().now();

        PCDstart_time = (pointcloud_receive_time - now_ros).seconds();
        dynamic_points.clear();
        dynamic_points_hash.clear();
        dynamic_cloud->clear();
        dynamic_points.reserve(dynamic_pcd_msg->points.size());
        for(const auto &dp : dynamic_pcd_msg->points)
        {
            Eigen::Vector3d pos(dp.position.x, dp.position.y, dp.position.z);
            Eigen::Vector3d vel(dp.velocity.x, dp.velocity.y, dp.velocity.z);
            dynamic_points.emplace_back(pos, vel);
            dynamic_points_hash[pos] = vel;
            pcl::PointXYZ pt{pos[0], pos[1], pos[2]};
            dynamic_cloud->points.push_back(pt);
        }
        _rrtPathPlanner.setInputDynamic(dynamic_points, _start_pos, PCDstart_time);
        if(dynamic_cloud->points.size() > 0)
        {
            dynamic_kdtree.setInputCloud(dynamic_cloud);
        }
    }
    void rcvPointCloudCallBack(const sensor_msgs::msg::PointCloud2::SharedPtr pointcloud_msg)
    {
        if (pointcloud_msg->data.empty())
            return;
        
        // Transform the point cloud from camera frame to map frame
        sensor_msgs::msg::PointCloud2 cloud_transformed;
        try
        {
            tf_buffer_.transform(*pointcloud_msg, cloud_transformed, "ground_link", tf2::durationFromSec(0.1));
        }
        catch (tf2::TransformException &ex)
        {
            RCLCPP_WARN(this->get_logger(), "Could not transform point cloud: %s", ex.what());
            return;
        }

        cloud_input.clear();
        pcl::fromROSMsg(cloud_transformed, cloud_input);

        if (cloud_input.points.empty())
            return;

        _is_has_map = true;
        _rrtPathPlanner.setInputStatic(cloud_input);
        
    }
    
    // Function to publish RRT waypoints
    void publishRRTWaypoints(const std::vector<Eigen::Vector3d>& path)
    {
        nav_msgs::msg::Path path_msg;
        path_msg.header.stamp = this->now();
        path_msg.header.frame_id = "ground_link";  // Adjust this frame to your use case

        for (const auto& point : path)
        {
            geometry_msgs::msg::PoseStamped pose;
            pose.header.stamp = this->now();
            pose.header.frame_id = "ground_link";
            pose.pose.position.x = point.x();
            pose.pose.position.y = point.y();
            pose.pose.position.z = point.z();
            path_msg.poses.push_back(pose);
        }

        _rrt_waypoints_pub->publish(path_msg);
    }

    void getCorridorPoints()
    {
        corridor_points.clear();
        corridor_points4d.clear();
        float r = 0;
        if(_rrtPathPlanner.getDis(_start_pos, _end_pos) < _commit_distance)
        {
            for (Eigen::Vector4d pt4d : _path_vector)
            {
                corridor_points.push_back(pt4d.head<3>());
                corridor_points4d.push_back(pt4d);
            }
            _corridor_end_pos = _end_pos;
            return;
        }
        else
        {
            for(int i=0; i<_path_vector.size(); i++)
            {
                if(r < _commit_distance && _radius[i] > _safety_margin)
                {
                    corridor_points.push_back(_path_vector[i].head<3>());
                    corridor_points4d.push_back(_path_vector[i]);
                    r += _radius[i];
                }
            }
            _corridor_end_pos = corridor_points[corridor_points.size()-1];
        }
    }
    
    std::vector<Eigen::Vector3d> getObstaclePoints(int &n, bool bkup = false)
    {
        std::vector<Eigen::Vector3d> obstacle_points;
        int k = cloud_input.points.size();
        for (int i=0; i<k; i++)
        {
            auto pt = cloud_input.points[i];
            Eigen::Vector3d pos(pt.x, pt.y, pt.z);
            obstacle_points.push_back(pos);
        }
        if(!bkup)
        {
            for (const auto& [position, velocity] : dynamic_points)
            {
                Eigen::Vector3d pos = position + n*delta_t*velocity;
                obstacle_points.push_back(pos);
            }
        }

        return obstacle_points;
    }

    Eigen::Matrix3Xd getObstaclePoints_continous(double &t1, double &t2, Eigen::Matrix<double, 6, 4> &bd)
    {
        Eigen::Matrix3Xd obstacle_points(3,0);
        int k = cloud_input.points.size();
        for (int i=0; i<k; i++)
        {
            auto pt = cloud_input.points[i];
            Eigen::Vector3d pos(pt.x, pt.y, pt.z);
            if ((bd.leftCols<3>() * pos + bd.rightCols<1>()).maxCoeff() < 0.0)
            {
                obstacle_points.conservativeResize(3, obstacle_points.cols() + 1);
                obstacle_points.col(obstacle_points.cols() - 1) = pos;
            }
        }
        // std::cout<<"input cloud size: "<<k<<std::endl;
        for (const auto& [position, velocity] : dynamic_points)
        {
            for(double t = t1; t <= t2; t += 0.2)
            {
                Eigen::Vector3d pos = position + (t-PCDstart_time)*velocity;
                if ((bd.leftCols<3>() * pos + bd.rightCols<1>()).maxCoeff() < 0.0)
                {
                    obstacle_points.conservativeResize(3, obstacle_points.cols() + 1);
                    obstacle_points.col(obstacle_points.cols() - 1) = pos;
                }
            }
        }

        return obstacle_points;
    }

    std::vector<Eigen::Vector3d> getObstaclePoints_display(double t)
    {
        std::vector<Eigen::Vector3d> obstacle_points;
        for (const auto& [position, velocity] : dynamic_points)
        {
            Eigen::Vector3d pos = position + t*velocity;
            obstacle_points.push_back(pos);
        }
        return obstacle_points;
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

    void convexCoverCIRI(const std::vector<Eigen::Vector3d> &path, 
        const double &range,
        std::vector<Eigen::MatrixX4d> &hpolys,
        const double eps, bool bkup = false)
    {
        Eigen::Vector3d lowCorner(-10.0, -30.0, 0.5);
        Eigen::Vector3d highCorner(70.0, 30.0, 5.0);
        hpolys.clear();
        int n = int(path.size());
        
        Eigen::Matrix<double, 6, 4> bd = Eigen::Matrix<double, 6, 4>::Zero();
        bd(0, 0) = 1.0;
        bd(1, 0) = -1.0;
        bd(2, 1) = 1.0;
        bd(3, 1) = -1.0;
        bd(4, 2) = 1.0;
        bd(5, 2) = -1.0;

        Eigen::MatrixX4d hp, gap;
        Eigen::MatrixX3d tangent_pcd1, tangent_pcd2;
        Eigen::Vector3d a(path[0][0], path[0][1], path[0][2]);
        Eigen::Vector3d b = a;
        std::vector<Eigen::Vector3d> valid_pc;
        std::vector<Eigen::Vector3d> bs;
        valid_pc.reserve(cloud_input.points.size());
        ciri.setupParams(_uav_radius, 4); // Setup CIRI with robot radius and iteration number
        for (int i = 1; i < n;)
        {

            Eigen::Vector3d path_point = path[i];
            a = b;
            b = path_point;
            i++;
            bs.emplace_back(b);
            bd(0, 3) = -std::min(std::max(a(0), b(0)) + range, highCorner(0));
            bd(1, 3) = +std::max(std::min(a(0), b(0)) - range, lowCorner(0));
            bd(2, 3) = -std::min(std::max(a(1), b(1)) + range, highCorner(1));
            bd(3, 3) = +std::max(std::min(a(1), b(1)) - range, lowCorner(1));
            bd(4, 3) = -std::min(std::max(a(2), b(2)) + range, highCorner(2));
            bd(5, 3) = +std::max(std::min(a(2), b(2)) - range, lowCorner(2));

            valid_pc.clear();
            auto obstacle_points = getObstaclePoints(n, bkup);

            for (const Eigen::Vector3d &p : obstacle_points)
            {
                if ((bd.leftCols<3>() * p + bd.rightCols<1>()).maxCoeff() < 0.0)
                {
                    valid_pc.emplace_back(p);
                }
            }
            if (valid_pc.empty()) {
                // std::cerr << "No valid points found for the current segment." << std::endl;
                Eigen::MatrixX4d temp_bp = bd;
                // firi::shrinkPolygon(temp_bp, _uav_radius);
                hpolys.emplace_back(temp_bp);
                continue;
            }

            Eigen::Map<const Eigen::Matrix<double, 3, -1, Eigen::ColMajor>> pc(valid_pc[0].data(), 3, valid_pc.size());

            if (ciri.convexDecomposition(bd, pc, a, b) != super_utils::SUCCESS) {
                std::cerr << "CIRI decomposition failed." << std::endl;
                hpolys.push_back(bd);
                continue;
            }
            pcd_origin = _start_pos;

            geometry_utils::Polytope optimized_poly;
            ciri.getPolytope(optimized_poly);
            hp = optimized_poly.GetPlanes(); // Assuming getPlanes() returns the planes of the polytope

            if (hpolys.size() != 0)
            {
                const Eigen::Vector4d ah(a(0), a(1), a(2), 1.0);
                if (3 <= ((hp * ah).array() > -eps).cast<int>().sum() +
                            ((hpolys.back() * ah).array() > -eps).cast<int>().sum())
                {
                    if (ciri.convexDecomposition(bd, pc, a, a) != super_utils::SUCCESS) 
                    {
                        std::cerr << "CIRI decomposition failed." << std::endl;
                        hpolys.push_back(bd);
                        continue;
                    }
                    ciri.getPolytope(optimized_poly);
                    gap = optimized_poly.GetPlanes(); // Assuming getPlanes() returns the planes of the polytope
                    hpolys.emplace_back(gap);
                }
            }
            hpolys.emplace_back(hp);
        }
    }

    void convexCoverCIRI_dynamic(const std::vector<Eigen::Vector4d> &path, 
        const double &range,
        std::vector<Eigen::MatrixX4d> &hpolys,
        const double eps)
    {
        Eigen::Vector3d lowCorner(_x_l, -20, _z_l2);
        Eigen::Vector3d highCorner(_x_h, 20, _z_h2);

        hpolys.clear();
        int n = int(path.size());
        
        Eigen::Matrix<double, 6, 4> bd = Eigen::Matrix<double, 6, 4>::Zero();
        bd(0, 0) = 1.0;
        bd(1, 0) = -1.0;
        bd(2, 1) = 1.0;
        bd(3, 1) = -1.0;
        bd(4, 2) = 1.0;
        bd(5, 2) = -1.0;

        Eigen::MatrixX4d hp, gap;
        Eigen::MatrixX3d tangent_pcd1, tangent_pcd2;
        Eigen::Vector3d a(path[0][0], path[0][1], path[0][2]);
        Eigen::Vector3d b = a;
        time_vector_poly.clear();
        std::vector<Eigen::Vector3d> valid_pc;
        std::vector<Eigen::Vector3d> bs;
        valid_pc.reserve(cloud_input.points.size());
        ciri.setupParams(_uav_radius, 4); // Setup CIRI with robot radius and iteration number
        for (int i = 1; i < n;)
        {
            Eigen::Vector3d path_point = path[i].head<3>();
            a = b;
            b = path_point;
            double t1 = 0.0;
            if(i != 0)
            {
                t1 = path[i-1][3];
            }
            double t2 = path[i][3];
            time_vector_poly.push_back(t2-t1);
            i++;
            bs.emplace_back(b);

            bd(0, 3) = -std::min(std::max(a(0), b(0)) + range, highCorner(0));
            bd(1, 3) = +std::max(std::min(a(0), b(0)) - range, lowCorner(0));
            bd(2, 3) = -std::min(std::max(a(1), b(1)) + range, highCorner(1));
            bd(3, 3) = +std::max(std::min(a(1), b(1)) - range, lowCorner(1));
            bd(4, 3) = -std::min(std::max(a(2), b(2)) + range, highCorner(2));
            bd(5, 3) = +std::max(std::min(a(2), b(2)) - range, lowCorner(2));

            valid_pc.clear();
            Eigen::Matrix3Xd obstacle_points = getObstaclePoints_continous(t1, t2, bd);
            if (obstacle_points.cols() == 0) {
                Eigen::MatrixX4d temp_bp = bd;
                hpolys.emplace_back(temp_bp);
                continue;
            }


            if (ciri.convexDecomposition(bd, obstacle_points, a, b) != super_utils::SUCCESS) {
                std::cerr << "CIRI decomposition failed." << std::endl;
                time_vector_poly.pop_back();
                continue;
            }
            pcd_origin = _start_pos;

            geometry_utils::Polytope optimized_poly;
            ciri.getPolytope(optimized_poly);
            hp = optimized_poly.GetPlanes(); // Assuming getPlanes() returns the planes of the polytope

            if (hpolys.size() != 0)
            {
                const Eigen::Vector4d ah(a(0), a(1), a(2), 1.0);
                if (3 <= ((hp * ah).array() > -eps).cast<int>().sum() +
                            ((hpolys.back() * ah).array() > -eps).cast<int>().sum())
                {
                    if (ciri.convexDecomposition(bd, obstacle_points, a, a) != super_utils::SUCCESS) 
                    {
                        std::cerr << "CIRI decomposition failed." << std::endl;
                        continue;
                    }
                    ciri.getPolytope(optimized_poly);
                    time_vector_poly.push_back(1.0);
                    gap = optimized_poly.GetPlanes(); // Assuming getPlanes() returns the planes of the polytope
                    hpolys.emplace_back(gap);
                }
            }
            hpolys.emplace_back(hp);
        }
    }

    void yaw_traj_generation(Trajectory<5> _traj, custom_interface_gym::msg::DesTrajectory &des_traj_msg)
    {
        double last_yaw = current_yaw;
        double tot_time = _traj.getTotalDuration();
        double dt_yaw = 0.1;
        int seg_num = ceil(tot_time / dt_yaw);
        dt_yaw = tot_time / seg_num;
        double forward_t = 0.1;

        std::vector<double> psi_vec;
        std::vector<int> wp_idx_vec;

        for (int i = 0; i < seg_num; i++) {
            double tc = i * dt_yaw;
            auto p1 = _traj.getPos(tc);
            double tf = std::min(tot_time, tc + forward_t);
            auto p2 = _traj.getPos(tf);
            auto pd = p2 - p1;

            double des_yaw = 0.0;
            if (pd.norm() > 1e-6) {
                des_yaw = atan2(pd[1], pd[0]);
                calcNextYaw(last_yaw, des_yaw);  // Ensure continuity
            } else if (!psi_vec.empty()) {
                des_yaw = psi_vec.back();
            }

            psi_vec.push_back(des_yaw);
            wp_idx_vec.push_back(i);
        }

        // Convert start and end yaw into control points using states2pts
        Eigen::Vector3d start_yaw_state, end_yaw_state;
        start_yaw_state << psi_vec.front(), 0.0, 0.0; // Assume yaw_dot = yaw_ddot = 0
        auto end_vel = _traj.getVel(tot_time - 0.1);
        double end_yaw = atan2(end_vel[1], end_vel[0]);
        calcNextYaw(last_yaw, end_yaw);
        end_yaw_state << end_yaw, 0.0, 0.0;

        Eigen::Matrix3d states2pts;
        states2pts << 
            1.0, -dt_yaw, (1.0 / 3.0) * dt_yaw * dt_yaw,
            1.0, 0.0,     -(1.0 / 6.0) * dt_yaw * dt_yaw,
            1.0, dt_yaw,  (1.0 / 3.0) * dt_yaw * dt_yaw;

        Eigen::Vector3d fixed_start = states2pts * start_yaw_state;
        Eigen::Vector3d fixed_end   = states2pts * end_yaw_state;

        // Number of control points = seg_num + 3 (cubic B-spline)
        int num_ctrl_pts = seg_num + 3;

        Eigen::MatrixXd yaw_control_points = yaw_opt.optimizeYawTrajCP(dt_yaw, psi_vec, wp_idx_vec, num_ctrl_pts, fixed_start, fixed_end);
        for(int i = 0; i<yaw_control_points.rows(); i++)
        {
            des_traj_msg.yaw_control_points.push_back(yaw_control_points(i,0));
        }
        des_traj_msg.yaw_interval = dt_yaw;
        des_traj_msg.yaw_enabled = des_traj_msg.YAW_ENABLED_TRUE;
    }

    void calcNextYaw(const double& last_yaw, double& yaw) 
    {
        // round yaw to [-PI, PI]
      
        double round_last = last_yaw;
      
        while (round_last < -M_PI) {
          round_last += 2 * M_PI;
        }
        while (round_last > M_PI) {
          round_last -= 2 * M_PI;
        }
      
        double diff = yaw - round_last;
      
        if (fabs(diff) <= M_PI) {
          yaw = last_yaw + diff;
        } else if (diff > M_PI) {
          yaw = last_yaw + diff - 2 * M_PI;
        } else if (diff < -M_PI) {
          yaw = last_yaw + diff + 2 * M_PI;
        }
    }

    void traj_generation_bkup(Eigen::Vector3d _traj_start_pos, Eigen::Vector3d _traj_start_vel, Eigen::Vector3d _traj_start_acc, Eigen::Vector3d _endpt, std::vector<Eigen::Matrix3d> dynamic_obs)
    {
        auto t1 = std::chrono::steady_clock::now();
        // GCopter parameters
        Eigen::Matrix3d iniState;
        Eigen::Matrix3d finState;
        iniState << _traj_start_pos, _traj_start_vel, _traj_start_acc;
        finState << _endpt, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();
        Eigen::VectorXd magnitudeBounds(5);
        Eigen::VectorXd penaltyWeights(5);
        Eigen::VectorXd physicalParams(6);
        std::vector<float> chiVec = {1000000, 10000, 10000, 10000, 100000, 1000};
        magnitudeBounds(0) = 1.0;
        magnitudeBounds(1) = 2.1;
        magnitudeBounds(2) = 1.05;
        magnitudeBounds(3) = 0.5*mass*9.8;
        magnitudeBounds(4) = 5*mass*9.8;
        penaltyWeights(0) = chiVec[0];
        penaltyWeights(1) = chiVec[1];
        penaltyWeights(2) = chiVec[2];
        penaltyWeights(3) = chiVec[3];
        penaltyWeights(4) = chiVec[4];
        // penaltyWeights(5) = chiVec[5];
        physicalParams(0) = mass;
        physicalParams(1) = 9.8;
        physicalParams(2) = horizontal_drag_coeff;
        physicalParams(3) = vertical_drag_coeff;
        physicalParams(4) = vertical_drag_coeff/10;
        physicalParams(5) = 0.0001;
        int quadratureRes = 16;
        float weightT = 20.0;
        float smoothingEps = 0.6;
        float relcostto1 = 1e-6;
        _traj.clear();
        auto t_now = rclcpp::Clock().now();
        double s_time = (t_now - pointcloud_receive_time).seconds();
        // !_gCopter.setup(weightT, iniState, finState, bkup_hpolys, INFINITY, smoothingEps, quadratureRes, magnitudeBounds, penaltyWeights, physicalParams)
        std::cout<<"matrix check backup traj gen 1"<<std::endl;
        if (!_gCopter.setup(weightT, iniState, finState, bkup_hpolys, INFINITY, smoothingEps, quadratureRes, magnitudeBounds, penaltyWeights, physicalParams))
        {
            std::cout<<"gcopter returned false during setup, traj exist set to false"<<std::endl;
            _is_traj_exist = false;
            custom_interface_gym::msg::DesTrajectory des_traj_msg;
            des_traj_msg.header.stamp = rclcpp::Clock().now();
            des_traj_msg.header.frame_id = "ground_link";
            des_traj_msg.action = des_traj_msg.ACTION_WARN_IMPOSSIBLE;
            _rrt_des_traj_pub->publish(des_traj_msg);
            return;
        }
        std::cout<<"matrix check backup traj gen 2"<<std::endl;

        if (std::isinf(_gCopter.optimize(_traj, relcostto1)))
        {
            std::cout<<"gcopter optimization cost is infinity, traj exist set to false"<<std::endl;
            _is_traj_exist = false;
            custom_interface_gym::msg::DesTrajectory des_traj_msg;
            des_traj_msg.header.stamp = rclcpp::Clock().now();
            des_traj_msg.header.frame_id = "ground_link";
            des_traj_msg.action = des_traj_msg.ACTION_WARN_IMPOSSIBLE;
            _rrt_des_traj_pub->publish(des_traj_msg);
            return;
        }
        std::cout<<"matrix check backup traj gen 3"<<std::endl;

        if (_traj.getPieceNum() > 0)
        {
            auto t2 = std::chrono::steady_clock::now();
            auto elapsed_traj = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()*0.001;
            traj_gen_time = elapsed_traj;
            bkup_trajstamp = rclcpp::Clock().now();
            _is_bkup_traj_exist = true;
            _is_traj_exist = false;
            custom_interface_gym::msg::DesTrajectory des_traj_msg;
            des_traj_msg.header.stamp = rclcpp::Clock().now();
            des_traj_msg.header.frame_id = "ground_link";
            des_traj_msg.trajectory_id = trajectory_id++;
            des_traj_msg.action = des_traj_msg.ACTION_ADD;
            des_traj_msg.num_order = order;
            std::cout<<"matrix check backup traj gen 4"<<std::endl;

            des_traj_msg.num_segment = _traj.getPieceNum();
            Eigen::VectorXd durations = _traj.getDurations();
            std::vector<double> durations_vec(durations.data(), durations.data() + durations.size());
            auto coefficient_mat = _traj.getCoefficientMatrices();
            std::cout<<"matrix check backup traj gen 5"<<std::endl;

            for(int i=0; i<_traj.getPieceNum(); i++)
            {
                des_traj_msg.duration_vector.push_back(durations_vec[i]);
                for(int j=0; j<coefficient_mat[i].rows(); j++)
                {
                    for(int k=0; k<coefficient_mat[i].cols(); k++)
                    {
                        des_traj_msg.matrices_flat.push_back(coefficient_mat[i](j, k));
                        // only for debugging 
                    }
                }
            }
            std::cout<<"matrix check backup traj gen 6"<<std::endl;

            des_traj_msg.debug_info = "trajectory_id: "+std::to_string(trajectory_id-1);

            // if(_is_yaw_enabled)
            // {
            //     yaw_traj_generation(_traj, des_traj_msg);
            // }
            std::cout<<"matrix check backup traj gen 7"<<std::endl;

            _rrt_des_traj_pub->publish(des_traj_msg);
            std::cout<<"matrix check backup traj gen 7"<<std::endl;

            // std::cout<<std::endl;
            return;
        }
    }

    void traj_generation(Eigen::Vector3d _traj_start_pos, Eigen::Vector3d _traj_start_vel, Eigen::Vector3d _traj_start_acc)
    {
        auto t1 = std::chrono::steady_clock::now();
        // GCopter parameters
        Eigen::Matrix3d iniState;
        Eigen::Matrix3d finState;
        iniState << _traj_start_pos, _traj_start_vel, _traj_start_acc;
        finState << _corridor_end_pos, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();
        Eigen::VectorXd magnitudeBounds(5);
        Eigen::VectorXd penaltyWeights(5);
        Eigen::VectorXd physicalParams(6);
        std::vector<float> chiVec = {10000, 10000, 10000, 10000, 100000};
        magnitudeBounds(0) = max_vel;
        magnitudeBounds(1) = 2.1;
        magnitudeBounds(2) = 1.05;
        magnitudeBounds(3) = 0.5*mass*9.8;
        magnitudeBounds(4) = t2w*mass*9.8;
        penaltyWeights(0) = chiVec[0];
        penaltyWeights(1) = chiVec[1];
        penaltyWeights(2) = chiVec[2];
        penaltyWeights(3) = chiVec[3];
        penaltyWeights(4) = chiVec[4];
        physicalParams(0) = mass;
        physicalParams(1) = 9.8;
        physicalParams(2) = horizontal_drag_coeff;
        physicalParams(3) = vertical_drag_coeff;
        physicalParams(4) = vertical_drag_coeff/10;
        physicalParams(5) = 0.0001;
        int quadratureRes = 16;
        float weightT = 20.0;
        float smoothingEps = 0.6;
        float relcostto1 = 1e-3;
        _traj.clear();
        if (!_gCopter.setup(weightT, iniState, finState, hpolys, INFINITY, smoothingEps, quadratureRes, magnitudeBounds, penaltyWeights, physicalParams))
        {
            std::cout<<"gcopter returned false during setup, traj exist set to false"<<std::endl;
            _is_traj_exist = false;
            custom_interface_gym::msg::DesTrajectory des_traj_msg;
            des_traj_msg.header.stamp = rclcpp::Clock().now();
            des_traj_msg.header.frame_id = "ground_link";
            des_traj_msg.action = des_traj_msg.ACTION_WARN_IMPOSSIBLE;
            _rrt_des_traj_pub->publish(des_traj_msg);
            return;
        }
        if (std::isinf(_gCopter.optimize(_traj, relcostto1)))
        {
            std::cout<<"gcopter optimization cost is infinity, traj exist set to false"<<std::endl;
            _is_traj_exist = false;
            custom_interface_gym::msg::DesTrajectory des_traj_msg;
            des_traj_msg.header.stamp = rclcpp::Clock().now();
            des_traj_msg.header.frame_id = "ground_link";
            des_traj_msg.action = des_traj_msg.ACTION_WARN_IMPOSSIBLE;
            _rrt_des_traj_pub->publish(des_traj_msg);
            return;
        }
        if (_traj.getPieceNum() > 0)
        {
            auto t2 = std::chrono::steady_clock::now();
            auto elapsed_traj = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()*0.001;
            traj_gen_time = elapsed_traj;
            trajstamp = rclcpp::Clock().now();
            _is_traj_exist = true;
            _is_bkup_traj_exist = false;
            custom_interface_gym::msg::DesTrajectory des_traj_msg;
            des_traj_msg.header.stamp = rclcpp::Clock().now();
            des_traj_msg.header.frame_id = "ground_link";
            des_traj_msg.trajectory_id = trajectory_id++;
            des_traj_msg.action = des_traj_msg.ACTION_ADD;
            des_traj_msg.num_order = order;
            des_traj_msg.num_segment = _traj.getPieceNum();
            Eigen::VectorXd durations = _traj.getDurations();
            std::vector<double> durations_vec(durations.data(), durations.data() + durations.size());
            auto coefficient_mat = _traj.getCoefficientMatrices();
            for(int i=0; i<_traj.getPieceNum(); i++)
            {
                des_traj_msg.duration_vector.push_back(durations_vec[i]);
                for(int j=0; j<coefficient_mat[i].rows(); j++)
                {
                    for(int k=0; k<coefficient_mat[i].cols(); k++)
                    {
                        des_traj_msg.matrices_flat.push_back(coefficient_mat[i](j, k));
                    }
                }
            }
            des_traj_msg.debug_info = "trajectory_id: "+std::to_string(trajectory_id-1);

            if(_is_yaw_enabled)
            {
                yaw_traj_generation(_traj, des_traj_msg);
            }
            _rrt_des_traj_pub->publish(des_traj_msg);
            double temp_commit_time = _traj.getTotalDuration()*commit_time;
            auto temp_commit_pos = _traj.getPos(temp_commit_time);
            Eigen::Vector4d root_coords = _rrtPathPlanner.getRootCoords();
            if((_start_pos - temp_commit_pos).norm() >= 1.0)
            {
                _commit_target.head<3>() = temp_commit_pos;
                _commit_target[3] = root_coords[3] + temp_commit_time;
            }
            else
            {
                _commit_target.head<3>() = _corridor_end_pos;
                _commit_target[3] = root_coords[3] + temp_commit_time/commit_time;
            }
            // std::cout<<std::endl;
            return;
        }
    }

    void traj_generation_fixed_time(Eigen::Vector3d _traj_start_pos, Eigen::Vector3d _traj_start_vel, Eigen::Vector3d _traj_start_acc, bool bkup = false)
    {
        auto t1 = std::chrono::steady_clock::now();
        // GCopter parameters
        Eigen::Matrix3d iniState;
        Eigen::Matrix3d finState;
        iniState << _traj_start_pos, _traj_start_vel, _traj_start_acc;
        std::vector<float> chiVec = {10000, 10000, 10000, 10000, 100000};

        if(bkup)
        {
            chiVec[0] = 100000;
            finState << bkup_goal, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();
        }
        else
        {
            finState << _corridor_end_pos, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();
        }
        Eigen::VectorXd magnitudeBounds(5);
        Eigen::VectorXd penaltyWeights(5);
        Eigen::VectorXd physicalParams(6);
        if(bkup)
        {
            magnitudeBounds(0) = 0.6 * max_vel;
        }
        else
        {
            magnitudeBounds(0) = max_vel;
        }
        magnitudeBounds(1) = 2.1;
        magnitudeBounds(2) = 1.05;
        magnitudeBounds(3) = 0.5*mass*9.8;
        magnitudeBounds(4) = t2w*mass*9.8;
        penaltyWeights(0) = chiVec[0];
        penaltyWeights(1) = chiVec[1];
        penaltyWeights(2) = chiVec[2];
        penaltyWeights(3) = chiVec[3];
        penaltyWeights(4) = chiVec[4];
        physicalParams(0) = mass;
        physicalParams(1) = 9.8;
        physicalParams(2) = horizontal_drag_coeff;
        physicalParams(3) = vertical_drag_coeff;
        physicalParams(4) = vertical_drag_coeff/10;
        physicalParams(5) = 0.0001;
        int quadratureRes = 16;
        float weightT = 20.0;
        float smoothingEps = 0.6;
        float relcostto1 = 1e-6;
        _traj.clear();
        Eigen::VectorXd time_vec_eig = Eigen::Map<Eigen::VectorXd>(time_vector_poly.data(), time_vector_poly.size());;
        // _gCopter_fixed.setup(iniState, finState, hpolys, INFINITY, smoothingEps, quadratureRes, magnitudeBounds, penaltyWeights, physicalParams, time_vec_eig)
        if(bkup)
        {
            if (!_gCopter_fixed.setup(iniState, finState, bkup_hpolys, INFINITY, smoothingEps, quadratureRes, magnitudeBounds, penaltyWeights, physicalParams, time_vec_eig))
            {
                std::cout<<"gcopter returned false during setup, traj exist set to false"<<std::endl;
                _is_traj_exist = false;
                custom_interface_gym::msg::DesTrajectory des_traj_msg;
                des_traj_msg.header.stamp = rclcpp::Clock().now();
                des_traj_msg.header.frame_id = "ground_link";
                des_traj_msg.action = des_traj_msg.ACTION_WARN_IMPOSSIBLE;
                _rrt_des_traj_pub->publish(des_traj_msg);
                return;
            }
        }
        else
        {
            if (!_gCopter_fixed.setup(iniState, finState, hpolys, INFINITY, smoothingEps, quadratureRes, magnitudeBounds, penaltyWeights, physicalParams, time_vec_eig))
            {
                std::cout<<"gcopter returned false during setup, traj exist set to false"<<std::endl;
                _is_traj_exist = false;
                custom_interface_gym::msg::DesTrajectory des_traj_msg;
                des_traj_msg.header.stamp = rclcpp::Clock().now();
                des_traj_msg.header.frame_id = "ground_link";
                des_traj_msg.action = des_traj_msg.ACTION_WARN_IMPOSSIBLE;
                _rrt_des_traj_pub->publish(des_traj_msg);
                return;
            }
        }
        if (std::isinf(_gCopter_fixed.optimize(_traj, relcostto1)))
        {
            std::cout<<"gcopter optimization cost is infinity, traj exist set to false"<<std::endl;
            _is_traj_exist = false;
            custom_interface_gym::msg::DesTrajectory des_traj_msg;
            des_traj_msg.header.stamp = rclcpp::Clock().now();
            des_traj_msg.header.frame_id = "ground_link";
            des_traj_msg.action = des_traj_msg.ACTION_WARN_IMPOSSIBLE;
            _rrt_des_traj_pub->publish(des_traj_msg);
            return;
        }
        if (_traj.getPieceNum() > 0)
        {
            if(!bkup)
            {
                Eigen::Vector3d temp_commit_target = _traj.getPos(_traj.getTotalDuration()*commit_time);
                if((temp_commit_target - _commit_target.head<3>()).norm() < _uav_radius)
                {
                    // std::cout<<"invalid commit target"<<std::endl;
                    _is_traj_exist = false;
                    current_state = INITIAL;
                    custom_interface_gym::msg::DesTrajectory des_traj_msg;
                    des_traj_msg.header.stamp = rclcpp::Clock().now();
                    des_traj_msg.header.frame_id = "ground_link";
                    des_traj_msg.action = des_traj_msg.ACTION_WARN_IMPOSSIBLE;
                    _rrt_des_traj_pub->publish(des_traj_msg);
                    return;
                }
                _commit_target.head<3>() = temp_commit_target;
                Eigen::Vector4d root_coords = _rrtPathPlanner.getRootCoords();
                _commit_target[3] = root_coords[3] + _traj.getTotalDuration()*commit_time;

                auto t2 = std::chrono::steady_clock::now();
                auto elapsed_traj = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()*0.001;
                traj_gen_time = elapsed_traj;
                auto t_curr = trajstamp;
                trajstamp = rclcpp::Clock().now();
                auto del_t = (t_curr - trajstamp).seconds();
                _is_traj_exist = true;
                _is_bkup_traj_exist = false;
            }
            else
            {
                auto t2 = std::chrono::steady_clock::now();
                auto elapsed_traj = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()*0.001;
                bkup_traj_gen_time = elapsed_traj;
                bkup_trajstamp = rclcpp::Clock().now();
                _is_traj_exist = false;
                _is_bkup_traj_exist = true;

            }
            custom_interface_gym::msg::DesTrajectory des_traj_msg;
            des_traj_msg.header.stamp = rclcpp::Clock().now();
            des_traj_msg.header.frame_id = "ground_link";
            des_traj_msg.trajectory_id = trajectory_id++;
            des_traj_msg.action = des_traj_msg.ACTION_ADD;
            des_traj_msg.num_order = order;
            des_traj_msg.num_segment = _traj.getPieceNum();
            Eigen::VectorXd durations = _traj.getDurations();
            std::vector<double> durations_vec(durations.data(), durations.data() + durations.size());
            auto coefficient_mat = _traj.getCoefficientMatrices();
            // for (auto mat : coefficient_mat)
            // {
            //     std::cout<<"######## mat #########"<<std::endl;
            //     std::cout<<mat<<std::endl;
            // }
            for(int i=0; i<_traj.getPieceNum(); i++)
            {
                des_traj_msg.duration_vector.push_back(durations_vec[i]);
                for(int j=0; j<coefficient_mat[i].rows(); j++)
                {
                    for(int k=0; k<coefficient_mat[i].cols(); k++)
                    {
                        des_traj_msg.matrices_flat.push_back(coefficient_mat[i](j, k));
                        // only for debugging 
                    }
                }
            }
            des_traj_msg.debug_info = "trajectory_id: "+std::to_string(trajectory_id-1);

            // if(!bkup)
            // {
            //     yaw_traj_generation(_traj, des_traj_msg);
            // }
            _rrt_des_traj_pub->publish(des_traj_msg);
            // std::cout<<std::endl;
            return;
        }
    }


    // Function to plan the initial trajectory using RRT
    void planInitialTraj()
    {
        _is_bkup_traj_exist = false;
        // std::cout<<"[Initial planning] in initial planning callback: "<<std::endl;
        _rrtPathPlanner.reset();
        _rrtPathPlanner.setPt(_start_pos, _end_pos, _x_l, _x_h, _y_l, _y_h, _z_l, _z_h,
                             _commit_distance, _max_samples, _sample_portion, _goal_portion, current_yaw, 0.6*max_vel, 0.75*max_vel, weight_t);
        init_planning_time = rclcpp::Clock().now();
        auto initial_start = std::chrono::steady_clock::now();
        double init_time = (init_planning_time - now_ros).seconds();
        _rrtPathPlanner.SafeRegionExpansion(0.4, init_time);
        std::tie(_path, _radius) = _rrtPathPlanner.getPath();
        if (_rrtPathPlanner.getPathExistStatus())
        {
            // Generate trajectory
            // std::cout<<"[Initial planning] initial path found: "<<_path.rows()<<std::endl;
            _path_vector = matrixToVector(_path);
            getCorridorPoints();
            auto t1 = std::chrono::steady_clock::now();
            convexCoverCIRI_dynamic(corridor_points4d, convexCoverRange, hpolys, 1.0e-6);
            // shortCutWithTimeIntervals();
            auto t2 = std::chrono::steady_clock::now();
            auto elapsed_convex = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()*0.001;

            convexDecompTime = elapsed_convex;
            // std::cout<<"[Initial planning] time taken in corridor generation "<<elapsed_convex<<std::endl;
            Eigen::Vector3d new_start_pos = _start_pos;
            Eigen::Vector3d new_start_vel = _start_vel;
            Eigen::Vector3d new_start_acc{0.0, 0.0, 0.0};
            if(_is_bkup_traj_exist)
            {
                auto t_curr = rclcpp::Clock().now();
                double del_t;
                del_t = (t_curr - bkup_trajstamp).seconds();
                if((del_t + 1.25*(convexDecompTime + traj_gen_time)) < _traj.getTotalDuration())
                {   
                    new_start_pos = _traj.getPos(del_t + 1.25*(convexDecompTime + traj_gen_time));
                    new_start_vel = _traj.getVel(del_t + 1.25*(convexDecompTime + traj_gen_time));
                    new_start_acc = _traj.getAcc(del_t + 1.25*(convexDecompTime + traj_gen_time));
                }
                _is_bkup_traj_exist = false;
            }
            traj_generation_fixed_time(new_start_pos, new_start_vel, new_start_acc);
            // traj_generation(new_start_pos, new_start_vel, new_start_acc);
            auto t3 = std::chrono::steady_clock::now();
            auto elapsed_gcopter = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()*0.001;

            if(_is_traj_exist)
            {
                auto initial_end = std::chrono::steady_clock::now();
                time_initial = std::chrono::duration_cast<std::chrono::duration<double>>(initial_end - initial_start).count();
                std::cout<<"initial planning complete time: "<<time_initial<<std::endl;
                std::cout<<"time taken in convex decomp: "<<elapsed_convex<<" and traj gen: "<<elapsed_gcopter<<std::endl;
                auto pathlist = _rrtPathPlanner.getPathList();
                int idx = newrootindex();

                // std::cout << "picked root_node: " << pathlist[idx]->coord.transpose() << std::endl;
                _rrtPathPlanner.resetRoot(idx);
                std::cout << "picked root_node(sanity check): " << _rrtPathPlanner.getRootCoords().transpose() << std::endl;

                visualizePolytope(hpolys);
                visualizeTrajectory(_traj, false);
            }      

        }
        else
        {
            RCLCPP_WARN(this->get_logger(), "No path found in initial trajectory planning");

            auto nlist = _rrtPathPlanner.getSelectedNodeList();
            // std::cout<<"[number of nodes] : "<<nlist.size()<<std::endl;
            _is_traj_exist = false;
            current_state = INITIAL;
        }
        visRrt(_rrtPathPlanner.getTree()); 
        visRRTPath(_path);
        visCommitTarget();
    }

    void planIncrementalTraj()
    {
        // if (_rrtPathPlanner.getGlobalNaviStatus())
        // {
        //     RCLCPP_WARN(this->get_logger(), "Almost reached final goal");
        //     sendFinalTrajectoryMessage();
        //     return;
        // }

        if (checkEndOfCommittedPath())
        {
            if (!_rrtPathPlanner.getPathExistStatus())
            {
                RCLCPP_WARN(this->get_logger(), "Reached committed target but no feasible path exists");
                _is_traj_exist = false;
                executeEmergencyStop();
                return;
            }
            else
            {
                auto incremental_start = std::chrono::steady_clock::now();

                auto t_curr = rclcpp::Clock().now();;
                auto del_t = (t_curr - trajstamp).seconds();
                std::tie(_path, _radius) = _rrtPathPlanner.getPath();
                _path_vector = matrixToVector(_path);
                getCorridorPoints();
                auto t1 = std::chrono::steady_clock::now();
                convexCoverCIRI_dynamic(corridor_points4d, convexCoverRange, hpolys, 1.0e-6);
                auto t2 = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()*0.001;

                // std::cout<<"[Incremental planner] reached committed target, time taken in corridor generation = "<<elapsed<<std::endl;
                Eigen::Vector3d new_traj_start_pos = _traj.getPos(_traj.getTotalDuration());
                Eigen::Vector3d new_traj_start_vel{0.0, 0.0, 0.0};
                Eigen::Vector3d new_traj_start_acc{0.0, 0.0, 0.0};
                // std::cout<<"[incremental planner] seg debug 0"<<std::endl;

                if(1.25*(convexDecompTime + traj_gen_time) < _traj.getTotalDuration() - del_t)
                {
                    new_traj_start_pos = _traj.getPos(del_t + 1.25*(convexDecompTime + traj_gen_time));
                    new_traj_start_vel = _traj.getVel(del_t + 1.25*(convexDecompTime + traj_gen_time));
                    new_traj_start_acc = _traj.getAcc(del_t + 1.25*(convexDecompTime + traj_gen_time));
                }
                // std::cout<<"[incremental planner] seg debug 1"<<std::endl;
                convexDecompTime = elapsed;

                traj_generation_fixed_time(new_traj_start_pos, new_traj_start_vel, new_traj_start_acc);
                auto t3 = std::chrono::steady_clock::now();
                auto elapsed_gcopter = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()*0.001;

                // traj_generation(new_traj_start_pos, new_traj_start_vel, new_traj_start_acc);

                // std::cout<<"[incremental planner] seg debug 2"<<std::endl;

                if(_is_traj_exist)
                {
                    auto incremental_end = std::chrono::steady_clock::now();
                    time_incremental = std::chrono::duration_cast<std::chrono::milliseconds>(incremental_end - incremental_start).count()*0.001;
                    std::cout<<"time spent in incremental planning: "<<time_incremental<<std::endl;
                    std::cout<<"time spent in convex_decomp: "<<convexDecompTime<<" gcopter: "<<elapsed_gcopter<<std::endl;
                    auto pathlist = _rrtPathPlanner.getPathList();
                    int idx = newrootindex();
                    _rrtPathPlanner.resetRoot(idx);
                }
                else
                {
                    RCLCPP_WARN(this->get_logger(), "Safe Trajectory could not be generated: Hovering");
                    custom_interface_gym::msg::DesTrajectory des_traj_msg;
                    des_traj_msg.header.stamp = rclcpp::Clock().now();
                    des_traj_msg.header.frame_id = "ground_link";
                    des_traj_msg.action = des_traj_msg.ACTION_WARN_IMPOSSIBLE;
                    _rrt_des_traj_pub->publish(des_traj_msg);
                }
                _path_vector = matrixToVector(_path);
                _radius_vector = radiusMatrixToVector(_radius);
            }
        }
        else
        {
            // std::cout<<"[Incremental planner] in refine and evaluate loop"<<std::endl;
            auto time_start_ref = std::chrono::steady_clock::now();
            // Continue refining and evaluating the path
            _rrtPathPlanner.SafeRegionRefine(0.3);
            _rrtPathPlanner.SafeRegionEvaluate(0.2);
            auto time_end_ref = std::chrono::steady_clock::now();

            // Get the updated path and publish it
            if(_rrtPathPlanner.getPathExistStatus())
            {
                // std::cout<<"[Incremental planner] in refine and evaluate loop: Path updated"<<std::endl;
                std::tie(_path, _radius) = _rrtPathPlanner.getPath();
                _path_vector = matrixToVector(_path);
            }
            double elapsed_ms = std::chrono::duration_cast<std::chrono::seconds>(time_end_ref - time_start_ref).count();
            // std::cout<<"[incremental planner] time duration: "<<elapsed_ms<<std::endl;
            visualizePolytope(hpolys);
            visualizeTrajectory(_traj, false);
        }
        //RCLCPP_DEBUG(this->get_logger(),"Traj updated");
        visRrt(_rrtPathPlanner.getTree());
        visRRTPath(_path); 
        visCommitTarget();

    }

    Eigen::Vector3d bkupDirection()
    {
        Eigen::Vector3d des_dir(0.0, 0.0, 0.0);
        double min_dist = INFINITY;
        int min_idx = 0;
        for(int i = 0; i < dynamic_obs_array.size(); i++)
        {
            auto obs_mat = dynamic_obs_array[i];
            Eigen::Vector3d pos_obs = obs_mat.row(0);
            Eigen::Vector3d vel_obs = obs_mat.row(1);
            Eigen::Vector3d shape_obs = obs_mat.row(2);
            
            double c = shape_obs[0];
            double b = shape_obs[1];
            double a = shape_obs[2];
            Eigen::Vector3d rel_pos = pos_obs - _start_pos; // r_o - r
            if(rel_pos.norm() < min_dist)
            {
                min_dist = rel_pos.norm();
                min_idx = i;
            }
            Eigen::Vector3d rel_vel = vel_obs - _start_vel;            
            double krep = 5.0;
            // Compute azimuth (phi) and elevation (theta) of rel_vel
            double rel_speed = rel_vel.norm();
            if (rel_speed < 1e-3) continue;

            // double theta = std::acos(rel_vel.z() / rel_speed); // elevation
            double theta   = std::atan2(rel_vel.y(), rel_vel.x()); // azimuth

            // Construct R matrix (rotation from global to ellipsoid-aligned frame)
            Eigen::Matrix2d R;
            // R << std::cos(phi)*std::cos(theta), -std::sin(phi),  std::cos(phi)*std::sin(theta),
            //     std::sin(phi)*std::cos(theta),  std::cos(phi),  std::sin(phi)*std::sin(theta),
            //     -std::sin(theta),               0,              std::cos(theta);
            R << std::cos(theta), -std::sin(theta),
            std::sin(theta), std::cos(theta);

            // Transform relative position into ellipsoid frame
            Eigen::Vector3d ellip_coords;
            ellip_coords.head<2>() = R * rel_pos.head<2>();
            double X = ellip_coords[0], Y = ellip_coords[1], Z = ellip_coords[2];
            double denom = std::pow(1 + (X*X)/(b*b) + (Y*Y)/(a*a), 2.0);

            // Force in ellipsoid-aligned frame (body frame)
            double Fx = krep * (-2.0 * X / (b*b)) / denom;
            double Fy = krep * (-2.0 * Y / (a*a)) / denom;
            double Fz = 0;

            Eigen::Vector3d F_ellipsoid(Fx, Fy, Fz);
            Eigen::Vector3d F_global;
            F_global.head<2>() = R.transpose() * F_ellipsoid.head<2>();
            des_dir += F_global;
        
        }
        des_dir = des_dir.normalized();
        return des_dir;
    }

    void planBackupTraj()
    {
        Eigen::Vector3d sp1;
        bkup_start_pos = _start_pos;
        Eigen::Vector3d des_dir(0.0, 0.0, 0.0);
        // std::cout<<"[bkup check] 2"<<std::endl;
        int min_idx = 0;
        bool obs_in = false;
        des_dir = bkupDirection();
        if(des_dir == Eigen::Vector3d::Zero())
        {
            des_dir = (_end_pos - _start_pos).normalized();
        }
        // std::cout<<"[bkup check] 1"<<std::endl;
        auto t1 = std::chrono::steady_clock::now();
        init_planning_time = rclcpp::Clock().now();
        double init_time = (init_planning_time - now_ros).seconds();
        std::vector<Eigen::MatrixX4d> large_hpolys;
        std::vector<Eigen::Vector3d> vec_pt;
        vec_pt.push_back(_start_pos), vec_pt.push_back(_start_pos);
        convexCoverCIRI(vec_pt, 3.0, large_hpolys, true);
        double lambda = ray_polygon_intersection(_start_pos, des_dir, large_hpolys[0]);
        bkup_goal = _start_pos + lambda * des_dir;
        // std::cout<<"[bkup check] bkup goal: "<<bkup_goal<<std::endl;
        Eigen::Vector3d new_traj_start_pos = _start_pos;
        Eigen::Vector3d new_traj_start_vel = Eigen::Vector3d::Zero();
        Eigen::Vector3d new_traj_start_acc = Eigen::Vector3d::Zero();
        // std::cout<<"[bkup check] 4"<<std::endl;
        
        _rrtPathPlanner.reset();
        // std::cout<<"[bkup check] 5"<<std::endl;
        // Compute bounding box of size 2.0 around _start_pos and bkup_goal
        Eigen::Vector3d min_pt = _start_pos.cwiseMin(bkup_goal);
        Eigen::Vector3d max_pt = _start_pos.cwiseMax(bkup_goal);

        // Expand by 1.0 in each direction (so box size = 2.0)
        double x_l_bkup = min_pt.x() - 1.0;
        double x_h_bkup = max_pt.x() + 1.0;
        double y_l_bkup = min_pt.y() - 1.0;
        double y_h_bkup = max_pt.y() + 1.0;

        // Now pass to planner
        _rrtPathPlanner.setPt(
            _start_pos, bkup_goal,
            x_l_bkup, x_h_bkup,
            y_l_bkup, y_h_bkup,
            _z_l, _z_h,
            _commit_distance, _max_samples, _sample_portion,
            _goal_portion, current_yaw, 0.5 * max_vel, 0.6 * max_vel, weight_t
        );
        
        _rrtPathPlanner.SafeRegionExpansion(1.5, init_time, true);
        auto bkup_nodelist = _rrtPathPlanner.getTree();
        std::tie(_path, _radius) = _rrtPathPlanner.getPath();

        if(_rrtPathPlanner.getPathExistStatus() && _path.rows() > 1)
        {
            bkup_hpolys.clear();
            time_vector_poly.clear();
            _path_vector = matrixToVector(_path);
            // std::cout<<"_path_vector size: "<<_path_vector.size()<<std::endl;
            convexCoverCIRI_dynamic(_path_vector, convexCoverRange, bkup_hpolys, 1.0e-6);
            // std::cout<<"BKUP_HPOLYS size: "<<bkup_hpolys.size()<<std::endl;
            // std::cout<<"time vector size: "<<time_vector_poly.size()<<std::endl;
            bkup_goal = _path_vector[_path_vector.size() - 1].head<3>();
        }
        else
        {
            // std::cout<< " -- [BACKUP PLANNING] ERROR! path not found, number of nodes added: "<<bkup_nodelist.size()<<std::endl;

            // std::cout<<"[Radius Start pt]: "<<_rrtPathPlanner.radiusSearch(bkup_nodelist[0]->coord)<<std::endl;
            time_vector_poly.clear();
            bkup_hpolys = large_hpolys;
            time_vector_poly.push_back((_start_pos - bkup_goal).norm()/(0.5 * max_vel));
        }
        // std::cout<<"[bkup check] 7"<<std::endl;

        auto t2 = std::chrono::steady_clock::now();
        bkup_convexDecompTime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()*0.001;
        visCommitTarget(true);
        // if(_server_active)
        // {
        //     double del_t = t_server;
        //     if(1.25*(bkup_convexDecompTime + bkup_traj_gen_time)< _traj.getTotalDuration() - del_t)
        //     {
        //         new_traj_start_pos = _traj.getPos(del_t + 1.25*(bkup_traj_gen_time + bkup_convexDecompTime));
        //         new_traj_start_vel = _traj.getVel(del_t + 1.25*(bkup_traj_gen_time + bkup_convexDecompTime));
        //         new_traj_start_acc = _traj.getAcc(del_t + 1.25*(bkup_traj_gen_time + bkup_convexDecompTime));
        //     }
        // }
        
        // std::cout<<"_start_pos: "<<_start_pos.transpose()<<std::endl;
        // std::cout<<"_bkup_goal: "<<bkup_goal.transpose()<<std::endl;
        t1 = std::chrono::steady_clock::now();
        traj_generation_fixed_time(new_traj_start_pos, new_traj_start_vel, new_traj_start_acc, true);
        t2 = std::chrono::steady_clock::now();
        bkup_traj_gen_time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()*0.001;

        // std::cout<<"[bkup check] 6"<<std::endl;
        if(_is_bkup_traj_exist)
        {
            visualizePolytope(bkup_hpolys);
            visualizeTrajectory(_traj, true);
        }
    }
    // Planning Callback (Core Path Planning Logic)
    
    void planningCallBack()
    {   
        if (!_is_target_receive || !_is_has_map)
        {
            RCLCPP_DEBUG(this->get_logger(), "No target or map received. Skipping planning cycle.");
            return;
        }

        if (_is_complete)
        {
            RCLCPP_WARN(this->get_logger(), "Reached GOAL");
            sendFinalTrajectoryMessage();
            // current_state = BACKUP;
            return;
        }

        if(force_test_backup)
        {
            current_state = BACKUP;
        }
        // ============================================================
        switch (current_state)
        {
            case INITIAL:
                // std::cout<<"initial case: "<<std::endl;
                planInitialTraj();
                if (_is_traj_exist)
                    current_state = INCREMENTAL;
                else
                {
                    if(near_dynamic)
                    {
                        current_state = BACKUP;
                    }
                    else
                    {
                        current_state = INITIAL;
                    }
                }
                break;

            case INCREMENTAL:
                planIncrementalTraj();
                if (!_is_traj_exist)
                {
                    if (near_dynamic) 
                    {
                        current_state = BACKUP;
                    } 
                    else 
                    {
                        current_state = INITIAL;
                    }    
                }
                break;

            case BACKUP:
            {
                if(disable_bkup)
                {
                    current_state = INITIAL;
                    return;
                }
                bool backup_expired = (_is_bkup_traj_exist && t_server >= _traj.getTotalDuration() * commit_time);
                bool need_replan_backup = false;
                if(_is_bkup_traj_exist && backup_expired && (near_dynamic)) need_replan_backup = true;
                if(!_is_bkup_traj_exist && (near_dynamic)) need_replan_backup = true;
                // std::cout<<"[Backup] need backup? "<<need_replan_backup<<" near dynamic obstacles? "<<near_dynamic<<std::endl;
                if(!need_replan_backup)
                {
                    current_state = INITIAL;    
                }
                if (need_replan_backup)
                {
                    num_bkup += 1;
                    std::cout<<"number of backup replans so far: "<<num_bkup<<std::endl;
                    // std::cout << "[Backup] Recomputing backup..." << std::endl;
                    planBackupTraj();
                    if (!_is_bkup_traj_exist)
                    {
                        RCLCPP_WARN(this->get_logger(), "Backup trajectory failed — Hovering");
                        executeEmergencyStop();
                    }
                    // current_state = INITIAL;
                }
                if (reachedBackupWaypoint())
                {
                    std::cout << "[Backup] Reached safe waypoint" << std::endl;
                    _is_bkup_traj_exist = false;
                    current_state = INITIAL;
                    // current_state = BACKUP;
                }
                break;
            }
        }  
    }

    double ray_polygon_intersection(Eigen::Vector3d origin, Eigen::Vector3d direction, Eigen::MatrixX4d poly)
    {
        direction = direction.normalized();
        double t_min = 0.0;
        double t_max = INFINITY;
        for(int i = 0; i<poly.rows(); i++)
        {
            Eigen::Vector3d n(poly(i, 0), poly(i, 1), poly(i, 2));
            double d = poly(i,3);
            double s = n.dot(direction);
            double o = n.dot(origin) + d;
            if(std::abs(s) < 1e-8)
            {
                if(o >= 0.0)
                {
                    std::cout<<"[ray intersection] INFINITY 1: "<<std::endl;
                    return INFINITY;
                }
                else
                {
                    continue;
                }
            }
            double t = -o/s;
            if(s > 0.0)
            {
                t_max = std::min(t_max, t);
            }
            else
            {
                t_min = std::max(t_min, t);
            }
        }
        if(t_min > t_max)
        {
            std::cout<<"[ray intersection] INFINITY 2: "<<std::endl;
            return INFINITY;
        }
        return t_max;
    }

    void sendFinalTrajectoryMessage() 
    {
        custom_interface_gym::msg::DesTrajectory des_traj_msg;
        des_traj_msg.header.stamp = rclcpp::Clock().now();
        des_traj_msg.header.frame_id = "ground_link";
        des_traj_msg.action = des_traj_msg.ACTION_WARN_FINAL;
        _rrt_des_traj_pub->publish(des_traj_msg);
    }

    void executeEmergencyStop() 
    {   
        RCLCPP_WARN(this->get_logger(), "Executing emergency stop");
        custom_interface_gym::msg::DesTrajectory des_traj_msg;
        des_traj_msg.header.stamp = rclcpp::Clock().now();
        des_traj_msg.header.frame_id = "ground_link";
        des_traj_msg.action = des_traj_msg.ACTION_WARN_IMPOSSIBLE;
        _rrt_des_traj_pub->publish(des_traj_msg);
        _is_traj_exist = false;
        _is_bkup_traj_exist = false;
    }


    int newrootindex()
    {
        double min_dist = INFINITY;
        auto plist = _rrtPathPlanner.getPathList();
        int idx = plist.size() - 1;
        // std::cout << "_commit_target: " << _commit_target.transpose() << std::endl;
        
        for (int i = plist.size() - 1; i >= 0; i--)
        {
            if (plist[i] != NULL)
            {
                double dist = (_commit_target.head<3>() - plist[i]->coord.head<3>()).norm();
                // std::cout << "plist[i] coord: " << plist[i]->coord.transpose()<<"index: "<<i<<" distance of plist[i] to node: "<<dist<< std::endl;
                if (dist < min_dist)
                {
                    min_dist = dist;
                    idx = i;
                }
            }
        }
        return idx;
    }

    void checkSafeTrajectory()
    {
        double t_since = 0.0;
        double t_since_start = 0.0;
        int dynamic_pcd_size = dynamic_cloud->points.size();

        if(_server_active && t_server != -1.0)
        {
            if(_is_bkup_traj_exist)
            {
                t_since = (bkup_trajstamp - pointcloud_receive_time).seconds();
                t_since_start = (bkup_trajstamp - now_ros).seconds();
            }
            else
            {
                t_since = (trajstamp - pointcloud_receive_time).seconds();
                t_since_start = (trajstamp - now_ros).seconds();
            }
            double traj_time = commit_time * _traj.getTotalDuration();
            // --- static obstacle check (keep your existing static check) ---
            for(double t = t_server + 1.0; t < traj_time; t += 0.1)
            {
                Eigen::Vector4d pos_t;
                pos_t.head<3>() = _traj.getPos(t);
                pos_t[3] = t + _rrtPathPlanner.getRootCoords()[3];
                pcl::PointXYZ searchPoint;
                searchPoint.x = pos_t(0);
                searchPoint.y = pos_t(1);
                searchPoint.z = pos_t(2);
                pointIdxRadiusSearch.clear();
                pointRadiusSquaredDistance.clear();
                static_kdtree.nearestKSearch(searchPoint, 1, pointIdxRadiusSearch, pointRadiusSquaredDistance);
                if (!pointRadiusSquaredDistance.empty()) {
                    double static_rad = sqrt(pointRadiusSquaredDistance[0]);
                    if(static_rad < _uav_radius)
                    {
                        visCollisionPoint(_traj.getPos(t));
                        executeEmergencyStop();
                        current_state = BACKUP;
                        return;
                    }
                }
            }
            
            if(dynamic_pcd_size == 0) return;

            double temp_t = commit_time * _traj.getTotalDuration() + 1.0;
            if(temp_t > _traj.getTotalDuration())
            {
                temp_t = _traj.getTotalDuration();
            }
            for(double t = t_server + 1.0; t < temp_t; t += 0.1)
            {
                Eigen::Vector4d pos_t;
                pos_t.head<3>() = _traj.getPos(t);
                pos_t[3] = t + t_since_start;
                double rad = _rrtPathPlanner.radiusSearch(pos_t, true);
                if(rad != -1.0 && rad < _uav_radius-_search_margin && _start_pos[0] < 20.0)
                {
                    if(_is_bkup_traj_exist && (t - (t_server + 1.0)) < 3.0) continue;
                    std::cout<<"################ COLLISION DETECTED UAV FOLLOWING TRAJ ################"<<std::endl;
                    std::cout<<"collision predicted at time: "<<t<<" in future: "<<std::endl;
                    std::cout<<"rad: "<<rad<<std::endl;

                    visCollisionPoint(pos_t.head<3>());
                    if(t - t_server < 3.0)
                    {
                        current_state = BACKUP;
                        executeEmergencyStop();
                        near_dynamic = true;
                        return;
                    }
                    else
                    {
                        current_state = INITIAL;
                        executeEmergencyStop();
                        return;
                    }
                }
            }
            near_dynamic = false;
        }
        else  if(!_is_bkup_traj_exist && !_is_traj_exist)
        {
            if(dynamic_pcd_size == 0) return;

            pcl::PointXYZ searchPoint;
            searchPoint.x = _start_pos(0);
            searchPoint.y = _start_pos(1);
            searchPoint.z = _start_pos(2);
            pointIdxRadiusSearch.clear();
            pointRadiusSquaredDistance.clear();

            int num_obstacles = 1;
            dynamic_kdtree.nearestKSearch(searchPoint, num_obstacles, pointIdxRadiusSearch, pointRadiusSquaredDistance);

            Eigen::Vector3d obs_pos(dynamic_cloud->points[pointIdxRadiusSearch[0]].x,
                                    dynamic_cloud->points[pointIdxRadiusSearch[0]].y,
                                    dynamic_cloud->points[pointIdxRadiusSearch[0]].z);

            Eigen::Vector3d rel_pos = obs_pos - _start_pos;
            if(rel_pos.norm() > 3.5) return;
            Eigen::Vector3d rel_vel = dynamic_points_hash[obs_pos] - _start_vel; // obstacle velocity

            double closing_rate = rel_pos.dot(rel_vel);
            if (closing_rate < 0) { // only if obstacle is moving towards UAV
                double rel_speed_sq = rel_vel.squaredNorm();
                if (rel_speed_sq > 1e-6) {
                    double t_closest = -closing_rate / rel_speed_sq;
                    if (t_closest >= 0) {
                        Eigen::Vector3d closest_point = rel_pos + t_closest * rel_vel;
                        double dist_closest = closest_point.norm();

                        if (dist_closest < 2.0 * _uav_radius) {
                            visCollisionPoint(_start_pos);
                            std::cout << " ###### COLLISION COURSE DETECTED ###### " << std::endl;
                            current_state = BACKUP;
                            near_dynamic = true;
                            return;
                        }
                    }
                }
            }
        }

    }

    bool reachedBackupWaypoint() 
    {
        Eigen::Vector3d curr_pos = _start_pos;  // or use odom if better
        double dist_to_bkup = (curr_pos - bkup_goal).norm();
        return dist_to_bkup < threshold;  // Threshold, tweak as needed
    }

    bool checkEndOfCommittedPath()
    {
        if(_is_target_arrive)
        {
            _is_target_arrive = false;
            return true;
        }
        else
        {
            return false;
        }
    }

    inline void visualizePolytope(const std::vector<Eigen::MatrixX4d> &hPolys)
    {
        visualization_msgs::msg::Marker mesh_marker;
        mesh_marker.header.frame_id = "ground_link";  // Replace with your desired frame ID
        mesh_marker.header.stamp = rclcpp::Clock().now();
        mesh_marker.ns = "polytope";
        mesh_marker.id = 0;  // Unique ID for the mesh
        mesh_marker.type = visualization_msgs::msg::Marker::TRIANGLE_LIST;  // Type: TRIANGLE_LIST
        mesh_marker.action = visualization_msgs::msg::Marker::ADD;

        mesh_marker.scale.x = 1.0;
        mesh_marker.scale.y = 1.0;
        mesh_marker.scale.z = 1.0;

        mesh_marker.color.r = 0.0f;  // Red
        mesh_marker.color.g = 1.0f;  // Green
        mesh_marker.color.b = 0.0f;  // Blue
        mesh_marker.color.a = 0.8f;  // Transparency

        // Marker for the wireframe (edges)
        visualization_msgs::msg::Marker edges_marker;
        edges_marker.header.frame_id = "ground_link";  // Same frame ID
        edges_marker.header.stamp = rclcpp::Clock().now();
        edges_marker.ns = "polytope_edges";
        edges_marker.id = 1;  // Unique ID for the edges
        edges_marker.type = visualization_msgs::msg::Marker::LINE_LIST;  // Type: LINE_LIST
        edges_marker.action = visualization_msgs::msg::Marker::ADD;

        edges_marker.scale.x = 0.02;  // Line thickness

        edges_marker.color.r = 1.0f;  // Red for edges
        edges_marker.color.g = 1.0f;  // Green for edges
        edges_marker.color.b = 1.0f;  // Blue for edges
        edges_marker.color.a = 1.0f;  // Full opacity

        // Iterate over polytopes
        for (const auto &hPoly : hPolys) {
            // Enumerate vertices of the polytope from half-space representation (Ax <= b)
            Eigen::Matrix<double, 3, -1, Eigen::ColMajor> vPoly;
            geo_utils::enumerateVs(hPoly, vPoly);  // Assumes `enumerateVs` computes vertices

            // Use QuickHull to compute the convex hull
            quickhull::QuickHull<double> tinyQH;
            const auto polyHull = tinyQH.getConvexHull(vPoly.data(), vPoly.cols(), false, true);
            const auto &idxBuffer = polyHull.getIndexBuffer();

            // Add triangles to the mesh marker
            for (size_t i = 0; i < idxBuffer.size(); i += 3) {
                geometry_msgs::msg::Point p1, p2, p3;

                // Vertex 1
                p1.x = vPoly(0, idxBuffer[i]);
                p1.y = vPoly(1, idxBuffer[i]);
                p1.z = vPoly(2, idxBuffer[i]);

                // Vertex 2
                p2.x = vPoly(0, idxBuffer[i + 1]);
                p2.y = vPoly(1, idxBuffer[i + 1]);
                p2.z = vPoly(2, idxBuffer[i + 1]);

                // Vertex 3
                p3.x = vPoly(0, idxBuffer[i + 2]);
                p3.y = vPoly(1, idxBuffer[i + 2]);
                p3.z = vPoly(2, idxBuffer[i + 2]);

                // Add points to the mesh marker
                mesh_marker.points.push_back(p1);
                mesh_marker.points.push_back(p2);
                mesh_marker.points.push_back(p3);

                // Add edges to the wireframe marker
                edges_marker.points.push_back(p1);
                edges_marker.points.push_back(p2);

                edges_marker.points.push_back(p2);
                edges_marker.points.push_back(p3);

                edges_marker.points.push_back(p3);
                edges_marker.points.push_back(p1);
            }
        }

        // Publish both markers
        _vis_mesh_pub->publish(mesh_marker);  // Publisher for the mesh
        _vis_edge_pub->publish(edges_marker);  // Publisher for the edges
    }

    void visualizeTrajectory(const Trajectory<5> &traj, bool bkup = false)
    {
        sensor_msgs::msg::PointCloud2 trajectory_cloud;
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr traj_points(new pcl::PointCloud<pcl::PointXYZRGBA>());


        double T = 0.01; // Sampling interval
        Eigen::Vector3d lastX = traj.getPos(0.0);

        for (double t = T; t < traj.getTotalDuration(); t += T) {
            Eigen::Vector3d X = traj.getPos(t);
            pcl::PointXYZRGBA point;

            // Add the current point to the trajectory point cloud
            point.x = X(0);
            point.y = X(1);
            point.z = X(2);
            if(!bkup)
            {
                point.r = 0;
                point.g = 255;
            }
            else
            {
                point.r = 255;
                point.g = 0;
            }
            point.b = 0;
            point.a = 255;
            traj_points->points.push_back(point);
        }
        pcl::toROSMsg(*traj_points, trajectory_cloud);

        // Set header information
        trajectory_cloud.header.frame_id = "ground_link";  // Replace "map" with your frame ID
        trajectory_cloud.header.stamp = rclcpp::Clock().now();
        _vis_trajectory_pub->publish(trajectory_cloud);

    }


    void visRrt(const std::vector<NodePtr_dynamic>& nodes)
    {
        visualization_msgs::msg::MarkerArray tree_markers;
        int marker_id = 0;

        // Get the tree from the RRT planner
        std::vector<NodePtr_dynamic> nodeList = _rrtPathPlanner.getTree();

        // Loop through all the nodes in the tree
        for (const auto &node : nodeList) {
            if (node->preNode_ptr != nullptr) { // Only visualize branches (paths)
                visualization_msgs::msg::Marker branch_marker;
                
                // Marker properties
                branch_marker.header.frame_id = "ground_link";
                branch_marker.header.stamp = this->get_clock()->now();
                branch_marker.ns = "rrt_branches";
                branch_marker.id = marker_id++;
                branch_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
                branch_marker.action = visualization_msgs::msg::Marker::ADD;

                // Define start and end points for the branch
                geometry_msgs::msg::Point start_point;
                start_point.x = node->coord[0];
                start_point.y = node->coord[1];
                start_point.z = node->coord[2];

                geometry_msgs::msg::Point end_point;
                end_point.x = node->preNode_ptr->coord[0];
                end_point.y = node->preNode_ptr->coord[1];
                end_point.z = node->preNode_ptr->coord[2];

                branch_marker.points.push_back(start_point);
                branch_marker.points.push_back(end_point);

                // Set branch properties: scale, color
                branch_marker.scale.x = 0.01; // Line width
                branch_marker.color.a = 0.8; // Transparency
                branch_marker.color.r = 0.0; // Red
                branch_marker.color.g = 0.0; // Green
                branch_marker.color.b = 1.0; // Blue (for branches)

                // Add the marker to the MarkerArray
                tree_markers.markers.push_back(branch_marker);
            }
        }

        // Publish the MarkerArray
        _vis_rrt_tree_pub->publish(tree_markers);

    }

    void visRRTPath(const Eigen::MatrixXd& path_matrix)
    {
        visualization_msgs::msg::MarkerArray path_visualizer;
        int marker_id = 0;
        
        visualization_msgs::msg::Marker point_vis_og;
            point_vis_og.header.frame_id = "ground_link";
            point_vis_og.header.stamp = this->get_clock()->now();
            point_vis_og.ns = "rrt_path";
            point_vis_og.id = marker_id++;
            point_vis_og.type = visualization_msgs::msg::Marker::LINE_STRIP;
            point_vis_og.action = visualization_msgs::msg::Marker::ADD;

            geometry_msgs::msg::Point p1, p2;
            p1.x = path_matrix(0,0);
            p1.y = path_matrix(0,1);
            p1.z = path_matrix(0,2);

            p2.x = path_matrix(1,0);
            p2.y = path_matrix(1,1);
            p2.z = path_matrix(1,2);
            point_vis_og.points.push_back(p1);
            point_vis_og.points.push_back(p2);
            point_vis_og.scale.x = 0.05; // Line width
            point_vis_og.color.a = 0.8; // Transparency
            point_vis_og.color.r = 1.0; // Red
            point_vis_og.color.g = 0.64; // Green
            point_vis_og.color.b = 0.0; // Blue (for branches)
            path_visualizer.markers.push_back(point_vis_og);

        for(int i=1; i < path_matrix.rows(); i++)
        {
            visualization_msgs::msg::Marker point_vis;
            point_vis.header.frame_id = "ground_link";
            point_vis.header.stamp = this->get_clock()->now();
            point_vis.ns = "rrt_path";
            point_vis.id = marker_id++;
            point_vis.type = visualization_msgs::msg::Marker::LINE_STRIP;
            point_vis.action = visualization_msgs::msg::Marker::ADD;

            geometry_msgs::msg::Point p1, p2;
            p1.x = path_matrix(i-1,0);
            p1.y = path_matrix(i-1,1);
            p1.z = path_matrix(i-1,2);

            p2.x = path_matrix(i,0);
            p2.y = path_matrix(i,1);
            p2.z = path_matrix(i,2);
            point_vis.points.push_back(p1);
            point_vis.points.push_back(p2);
            point_vis.scale.x = 0.05; // Line width
            point_vis.color.a = 0.8; // Transparency
            point_vis.color.r = 1.0; // Red
            point_vis.color.g = 0.64; // Green
            point_vis.color.b = 0.0; // Blue (for branches)

            path_visualizer.markers.push_back(point_vis);
        }
        _vis_rrt_path_pub->publish(path_visualizer);
    }

    void visCollisionPoint(Eigen::Vector3d col_pt)
    {
        
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = "ground_link";
            marker.header.stamp = this->now();
            marker.ns = "corridor";
            marker.id = 0;
            marker.type = visualization_msgs::msg::Marker::SPHERE;
            marker.action = visualization_msgs::msg::Marker::ADD;
            
            // Set position of the marker
            
            marker.pose.position.x = col_pt(0);
            marker.pose.position.y = col_pt(1);
            marker.pose.position.z = col_pt(2);
        
            
            // Set scale (diameter based on radius)
            double diameter = 2.0 * 0.25; // Radius to diameter
            marker.scale.x = diameter;
            marker.scale.y = diameter;
            marker.scale.z = diameter;

            // Set color and transparency
            marker.color.a = 0.5;  // Transparency
            marker.color.r = 1.0;
            marker.color.g = 0.0;
            marker.color.b = 0.0;

            _vis_collision_point->publish(marker);
    }

    void visCommitTarget(bool bkup = false)
    {
        
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = "ground_link";
            marker.header.stamp = this->now();
            marker.ns = "corridor";
            marker.id = 0;
            marker.type = visualization_msgs::msg::Marker::SPHERE;
            marker.action = visualization_msgs::msg::Marker::ADD;
            
            // Set position of the marker
            if(bkup)
            {
                marker.pose.position.x = bkup_goal(0);
                marker.pose.position.y = bkup_goal(1);
                marker.pose.position.z = bkup_goal(2);
            }
            else
            {
                marker.pose.position.x = _commit_target(0);
                marker.pose.position.y = _commit_target(1);
                marker.pose.position.z = _commit_target(2);
            }
            
            // Set scale (diameter based on radius)
            double diameter = 2.0 * 0.25; // Radius to diameter
            marker.scale.x = diameter;
            marker.scale.y = diameter;
            marker.scale.z = diameter;

            // Set color and transparency
            marker.color.a = 0.5;  // Transparency
            if(bkup)
            {
                marker.color.r = 1.0;
                marker.color.g = 0.0;
            }
            else
            {
                marker.color.r = 0.0;
                marker.color.g = 1.0;
            }
            marker.color.b = 0.0;

            _vis_commit_target->publish(marker);
    }

    void visualizeObs(const std::vector<geometry_utils::Ellipsoid> &tangent_obs, int id)
    {

        visualization_msgs::msg::MarkerArray marker_array;
        
        for (size_t i = 0; i < tangent_obs.size(); ++i)
        {
            const auto &obs = tangent_obs[i];
            auto center = obs.d() - pcd_origin;
            auto axes = obs.r();
            
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = "map";
            marker.header.stamp = this->get_clock()->now();
            marker.ns = "ellipsoids";
            marker.id = id * 100 + i;
            marker.type = visualization_msgs::msg::Marker::SPHERE;
            marker.action = visualization_msgs::msg::Marker::ADD;
            
            marker.pose.position.x = center[0];
            marker.pose.position.y = center[1];
            marker.pose.position.z = center[2];
            marker.pose.orientation.w = 1.0; // Identity rotation
            
            marker.scale.x = 2.0 * axes[0]; // Major axis
            marker.scale.y = 2.0 * axes[1]; // Minor axis
            marker.scale.z = 2.0 * axes[2]; // Minor axis
            
            double r = 1.0;
            double g = 0.0;
            double b = 0.0;
            
            marker.color.r = r;
            marker.color.g = g;
            marker.color.b = b;
            marker.color.a = 1.0; // Transparent ellipsoids
            
            marker.lifetime = rclcpp::Duration::from_seconds(0);
            marker_array.markers.push_back(marker);
        }
        
        _vis_ellipsoid->publish(marker_array);
        // std::cout<<"[obstacle vis debug] obstacle visualized"<<std::endl;
    }

    std::vector<Eigen::Vector4d> matrixToVector(const Eigen::MatrixXd& path_matrix)
    {
        std::vector<Eigen::Vector4d> path_vector;
        for (int i = 0; i < path_matrix.rows(); ++i)
        {
            Eigen::Vector4d point;
            point[0] = path_matrix(i, 0);
            point[1] = path_matrix(i, 1);
            point[2] = path_matrix(i, 2);
            point[3] = path_matrix(i, 3);
            path_vector.push_back(point);
        }
        return path_vector;
    }

    std::vector<double> radiusMatrixToVector(const Eigen::Matrix<double, -1, 1>& eigen_matrix)
    {
        std::vector<double> vec(eigen_matrix.data(), eigen_matrix.data() + eigen_matrix.size());
        return vec;
    }

};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PointCloudPlanner>());
    rclcpp::shutdown();
    return 0;
}