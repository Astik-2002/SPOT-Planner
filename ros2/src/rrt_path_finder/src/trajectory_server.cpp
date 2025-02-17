#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "custom_interface_gym/msg/des_trajectory.hpp"
#include "custom_interface_gym/msg/traj_msg.hpp"
#include <nav_msgs/msg/path.hpp>
#include "geometry_msgs/msg/pose_stamped.hpp"

#include "rrt_path_finder/firi.hpp"
#include "rrt_path_finder/gcopter.hpp"
#include "rrt_path_finder/trajectory.hpp"
#include "rrt_path_finder/geo_utils.hpp"
#include "rrt_path_finder/quickhull.hpp"

#include <Eigen/Eigen>
#include <vector>
#include <iostream>

constexpr int D = 5; // Polynomial order D (for example, 5)

class TrajectoryServer : public rclcpp::Node
{
private:
    rclcpp::Subscription<custom_interface_gym::msg::DesTrajectory>::SharedPtr trajectory_sub;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_sub;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr dest_pts_sub;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr yaw_target_sub;
    rclcpp::Publisher<custom_interface_gym::msg::TrajMsg>::SharedPtr command_pub;
    rclcpp::TimerBase::SharedPtr command_timer;

    // Store current trajectory
    std::vector<Eigen::Matrix<double, 3, 6>> current_coefficients;  // Vector of (3, D+1) matrices for each trajectory segment
    std::vector<double> segment_durations;
    Eigen::Vector3d current_pos{-2.0, 0.0, 1.5};
    Eigen::Vector3d end_pos, yaw_target;
    bool _is_target_receive = false;
    bool _is_goal_arrive = false;
    bool _abort_hover_set = false;
    int num_segments;
    int order = D+1;
    Trajectory<5> _traj;
    int trajectory_id = 0;
    rclcpp::Time trajectory_start_time;
    bool has_trajectory;
    bool is_aborted;
    bool hover_command_sent = false;
    nav_msgs::msg::Odometry _odom;
    rclcpp::Time _final_time = rclcpp::Time(0);
    rclcpp::Time _start_time = rclcpp::Time::max();
    
    double previous_yaw_ = 0.0; // Store the last yaw value
    double dyaw_filtered_ = 0.0; // Filtered yaw rate
    struct Parameters {
        double dc = 0.01;                // Time step for yaw update
        double w_max = 1.0;            // Maximum yaw rate
        double alpha_filter_dyaw = 0.5; // Smoothing factor for dyaw
    } par_;

    bool face_yaw_goal = false; // Flag to determine yaw behavior


public:
    TrajectoryServer()
        : Node("trajectory_server"),
          has_trajectory(false),
          is_aborted(false)
    {
        // Initialize subscribers and publishers
        trajectory_sub = this->create_subscription<custom_interface_gym::msg::DesTrajectory>(
            "des_trajectory", 10, std::bind(&TrajectoryServer::trajectoryCallback, this, std::placeholders::_1));
        
        odometry_sub = this->create_subscription<nav_msgs::msg::Odometry>(
            "odom", 10, std::bind(&TrajectoryServer::rcvOdomCallback, this, std::placeholders::_1));

        dest_pts_sub = this->create_subscription<nav_msgs::msg::Path>(
            "waypoints", 1, std::bind(&TrajectoryServer::rcvGoalCallback, this, std::placeholders::_1));

        yaw_target_sub = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "corridor_endpoint", 1, std::bind(&TrajectoryServer::rcvYawCallback, this, std::placeholders::_1));

        command_pub = this->create_publisher<custom_interface_gym::msg::TrajMsg>("rrt_command",10);
        // Timer for periodic command updates
        command_timer = this->create_wall_timer(
            std::chrono::milliseconds(10), std::bind(&TrajectoryServer::commandCallback, this));
    }

    void rcvOdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        _odom = *msg;
        current_pos[0] = _odom.pose.pose.position.x;
        current_pos[1] = _odom.pose.pose.position.y;
        current_pos[2] = _odom.pose.pose.position.z;

        auto& orientation = _odom.pose.pose.orientation;
        double siny_cosp = 2.0 * (orientation.w * orientation.z + orientation.x * orientation.y);
        double cosy_cosp = 1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z);
        double yaw = std::atan2(siny_cosp, cosy_cosp);

        // Update previous_yaw_ with the current yaw
        previous_yaw_ = yaw;
        // RCLCPP_WARN(this->get_logger(), "Received odometry: position(x: %.2f, y: %.2f, z: %.2f)",
               // _odom.pose.pose.position.x, _odom.pose.pose.position.y, _odom.pose);
    }

    void rcvGoalCallback(const nav_msgs::msg::Path::SharedPtr wp_msg)
    {
        if(_is_target_receive) return;
        if (wp_msg->poses.empty() || wp_msg->poses[0].pose.position.z < 0.0)
            return;

        end_pos(0) = wp_msg->poses[0].pose.position.x;
        end_pos(1) = wp_msg->poses[0].pose.position.y;
        end_pos(2) = wp_msg->poses[0].pose.position.z;
        
        _is_target_receive = true;
    }

    void rcvYawCallback(const geometry_msgs::msg::PoseStamped::SharedPtr yaw_msg)
    {
        yaw_target(0) = yaw_msg->pose.position.x;
        yaw_target(1) = yaw_msg->pose.position.y;
        yaw_target(2) = yaw_msg->pose.position.z;
        std::cout<<"[yaw debug]: yaw target received"<<std::endl; 
        
    }

    void trajectoryCallback(const custom_interface_gym::msg::DesTrajectory::SharedPtr msg)
    {
        std::cout<<"in trajectory callback"<<std::endl;

        switch (msg->action)
        {
        case custom_interface_gym::msg::DesTrajectory::ACTION_ADD:
            std::cout<<"case Add"<<std::endl;
            handleAddTrajectory(msg);
            break;
        case custom_interface_gym::msg::DesTrajectory::ACTION_BACKUP:
            std::cout<<"case backup"<<std::endl;
            handleBackupTrajectory(msg);
            break;
        case custom_interface_gym::msg::DesTrajectory::ACTION_WARN_FINAL:
            std::cout<<"case Final"<<std::endl;
            handleFinalTrajectory();
            break;
        default:
            handleAbortTrajectory();
            RCLCPP_ERROR(this->get_logger(), "action command received: %u", msg->action);

        }
    }

    void handleAddTrajectory(const custom_interface_gym::msg::DesTrajectory::SharedPtr msg)
    {
        if(msg->trajectory_id < trajectory_id)
        {
            std::cout<<"backward trajectory invalid"<<std::endl;
            return;
        }
        std::cout<<"in handle add trajectory callback"<<std::endl;
        has_trajectory = true;
        is_aborted = false;
        hover_command_sent = false;
        _traj.clear();
        trajectory_id = msg->trajectory_id;
        _start_time = msg->header.stamp;
        _final_time = _start_time;
        segment_durations.clear();
        current_coefficients.clear();
        segment_durations = msg->duration_vector;
        num_segments = msg->num_segment;
        order = msg->num_order;

        std::vector<double> array_msg_traj;
        array_msg_traj = msg->matrices_flat;

        for (int i = 0; i < segment_durations.size(); ++i) 
        {
            Eigen::Map<const Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> matrix(
            array_msg_traj.data() + i * 3 * 6); // Starting point in the vector

            // Store the matrix in the output vector
            current_coefficients.push_back(matrix);

            std::chrono::duration<double> duration_in_sec(segment_durations[i]);
            rclcpp::Duration rcl_duration(duration_in_sec);

        }
        // for (auto mat : current_coefficients)
        // {
        //     std::cout<<"######## mat #########"<<std::endl;
        //     std::cout<<mat<<std::endl;
        // }
        _traj.setParameters(segment_durations, current_coefficients);
        std::cout<<"in handle add trajectory callback, traj set successfully"<<std::endl;

        // RCLCPP_WARN(this->get_logger(), "Added trajectory with %lu segments.", msg->num_segment);
        
    }

    void handleBackupTrajectory(const custom_interface_gym::msg::DesTrajectory::SharedPtr msg)
    {
        std::cout<<"in handle backup trajectory callback"<<std::endl;
        has_trajectory = true;
        is_aborted = false;
        hover_command_sent = false;
        _traj.clear();
        trajectory_id = msg->trajectory_id;
        _start_time = msg->header.stamp;
        _final_time = _start_time;
        segment_durations.clear();
        current_coefficients.clear();
        segment_durations = msg->duration_vector;
        num_segments = msg->num_segment;
        order = msg->num_order;

        std::vector<double> array_msg_traj;
        array_msg_traj = msg->matrices_flat;

        for (int i = 0; i < segment_durations.size(); ++i) 
        {
            Eigen::Map<const Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> matrix(
            array_msg_traj.data() + i * 3 * 6); // Starting point in the vector

            // Store the matrix in the output vector
            current_coefficients.push_back(matrix);

            std::chrono::duration<double> duration_in_sec(segment_durations[i]);
            rclcpp::Duration rcl_duration(duration_in_sec);

        }
        _traj.setParameters(segment_durations, current_coefficients);
        std::cout<<"in handle backup trajectory callback, traj set successfully"<<std::endl;
        
    }

    void handleAbortTrajectory()
    {
        has_trajectory = false;
        is_aborted = true;
        RCLCPP_WARN(this->get_logger(), "Trajectory aborted.");
    }

    void handleFinalTrajectory()
    {
        _is_goal_arrive = true;
    }

    void saturate(double& value, double min_val, double max_val)
    {
        value = std::min(std::max(value, min_val), max_val);
    }

    void angle_wrap(double& angle)
    {
        while (angle > M_PI) angle -= 2 * M_PI;
        while (angle < -M_PI) angle += 2 * M_PI;
    }

    void yaw(double diff, custom_interface_gym::msg::TrajMsg& next_goal)
    {
        saturate(diff, -par_.dc * par_.w_max, par_.dc * par_.w_max);
        double dyaw_not_filtered = copysign(1.0, diff) * par_.w_max;

        dyaw_filtered_ = (1 - par_.alpha_filter_dyaw) * dyaw_not_filtered +
                         par_.alpha_filter_dyaw * dyaw_filtered_;
        next_goal.yaw = previous_yaw_ + dyaw_filtered_ * par_.dc;
    }

    void getDesiredYaw(custom_interface_gym::msg::TrajMsg& next_goal, const Eigen::Vector3d& current_pos, const Eigen::Vector3d& goal_pos, const Eigen::Vector3d& heading_pos)
    {
        // Face the goal
        // double desired_yaw = atan2(goal_pos.y() - current_pos.y(),
        //                     goal_pos.x() - current_pos.x());
        double desired_yaw = 0.0;
        if (face_yaw_goal)
        {
            // Face the goal
            desired_yaw = atan2(goal_pos.y() - current_pos.y(),
                                goal_pos.x() - current_pos.x());
        }
        else
        {
            // Face the direction of travel
            Eigen::Vector3d direction = heading_pos - current_pos;
            desired_yaw = atan2(direction.y(), direction.x());
        }
        double diff = desired_yaw - previous_yaw_;
        angle_wrap(diff);

        yaw(diff, next_goal);
    }

    void commandCallback()
    {
        if (!has_trajectory || is_aborted)
        {
            if(hover_command_sent) return;
            else
            {
                hover_command_sent = true;
                custom_interface_gym::msg::TrajMsg traj_msg;
                traj_msg.header.stamp = rclcpp::Clock().now();
                traj_msg.header.frame_id = "ground_link"; 

                traj_msg.hover = true;
                // Publish the message
                command_pub->publish(traj_msg); // Replace traj_publisher_ with your actual publisher variable

            }
            return;
        }

        if (_is_goal_arrive)
        {
            custom_interface_gym::msg::TrajMsg traj_msg;
            traj_msg.header.stamp = rclcpp::Clock().now();
            traj_msg.header.frame_id = "ground_link"; 

            // Set position
            traj_msg.position.x = end_pos[0];
            traj_msg.position.y = end_pos[1];
            traj_msg.position.z = end_pos[2];
            std::cout<<"[Goal setting] current position: "<<current_pos[0]<<":"<<current_pos[1]<<":"<<current_pos[2]<<std::endl;
            std::cout<<"[Goal setting] command position: "<<traj_msg.position.x<<":"<<traj_msg.position.y<<":"<<traj_msg.position.z<<std::endl;

            // Publish the message
            traj_msg.hover = true;
            command_pub->publish(traj_msg); // Replace traj_publisher_ with your actual publisher variable

            return;
        }
        // Get elapsed time
        double elapsed = (this->get_clock()->now() - _start_time).seconds();
        Eigen::Vector3d des_pos;
        Eigen::Vector3d des_vel;
        Eigen::Vector3d des_Acc;
        Eigen::Vector3d des_jerk;
        
        if (elapsed > _traj.getTotalDuration())
        {
            // RCLCPP_WARN_THROTTLE(this->get_logger(), 1.0, "Trajectory completed. Waiting for new trajectory...");
            des_pos = _traj.getPos(_traj.getTotalDuration());
            des_vel = _traj.getVel(_traj.getTotalDuration());
            des_Acc = _traj.getAcc(_traj.getTotalDuration());
            des_jerk = _traj.getJer(_traj.getTotalDuration());
        }
        else
        {
            des_pos = _traj.getPos(elapsed);
            des_vel = _traj.getVel(elapsed);
            des_Acc = _traj.getAcc(elapsed);
            des_jerk = _traj.getJer(elapsed);
        }

        // std::cout<<"elapsed: "<<elapsed<<std::endl;
        custom_interface_gym::msg::TrajMsg traj_msg;
        traj_msg.header.stamp = rclcpp::Clock().now();
        traj_msg.header.frame_id = "ground_link"; 

        // Set position
        traj_msg.position.x = des_pos.x();
        traj_msg.position.y = des_pos.y();
        traj_msg.position.z = des_pos.z();

        // Set velocity
        traj_msg.velocity.x = des_vel.x();
        traj_msg.velocity.y = des_vel.y();
        traj_msg.velocity.z = des_vel.z();

        // Set acceleration
        traj_msg.acceleration.x = des_Acc.x();
        traj_msg.acceleration.y = des_Acc.y();
        traj_msg.acceleration.z = des_Acc.z();

        // Set jerk
        traj_msg.jerk.x = des_jerk.x();
        traj_msg.jerk.y = des_jerk.y();
        traj_msg.jerk.z = des_jerk.z();

        // Set yaw 
        getDesiredYaw(traj_msg, current_pos, yaw_target, des_pos);
        // traj_msg.yaw = 0;
        // std::cout<<"[Traj follow] error in position: "<<(current_pos - des_pos).norm()<<std::endl;

        // Publish the message
        command_pub->publish(traj_msg); // Replace traj_publisher_ with your actual publisher variable
    }
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TrajectoryServer>());
    rclcpp::shutdown();
    return 0;
}
