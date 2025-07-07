#include <rclcpp/rclcpp.hpp>
#include <mavros_msgs/msg/position_target.hpp>
#include <mavros_msgs/msg/state.hpp>
#include <mavros_msgs/srv/command_bool.hpp>
#include <mavros_msgs/srv/set_mode.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <custom_interface_gym/msg/des_trajectory.hpp>
#include <chrono>
#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include <limits>
#include <nav_msgs/msg/path.hpp>
#include "nav_msgs/msg/odometry.hpp"
#include "rrt_path_finder/firi.hpp"
#include "rrt_path_finder/gcopter.hpp"
#include "rrt_path_finder/trajectory.hpp"
#include "rrt_path_finder/geo_utils.hpp"
#include "rrt_path_finder/quickhull.hpp"
#include "rrt_path_finder/non_uniform_bspline.hpp"
#include <Eigen/Eigen>
#include <cmath>
#include <algorithm>
#include <memory>

constexpr int D = 5;
using namespace std::chrono_literals;

class OffboardControl : public rclcpp::Node
{
public:
  OffboardControl() : Node("offboard_control")
  {
    rclcpp::QoS mavros_qos(10);
    mavros_qos.best_effort();  // Match MAVROS default
    mavros_qos.durability_volatile();

    // Publishers
    setpoint_raw_pub_ = this->create_publisher<mavros_msgs::msg::PositionTarget>(
        "/mavros/setpoint_raw/local", mavros_qos);
    marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
        "/drone_marker", mavros_qos);

    // Subscribers
    state_sub_ = this->create_subscription<mavros_msgs::msg::State>(
        "/mavros/state", mavros_qos,
        std::bind(&OffboardControl::stateCallback, this, std::placeholders::_1));
    local_pos_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        "/mavros/local_position/pose", mavros_qos,
        std::bind(&OffboardControl::localPositionCallback, this, std::placeholders::_1));

    trajectory_sub_ = this->create_subscription<custom_interface_gym::msg::DesTrajectory>(
        "/des_trajectory", 1,
        std::bind(&OffboardControl::trajectoryCallback, this, std::placeholders::_1));
    _dest_pts_sub = this->create_subscription<nav_msgs::msg::Path>(
        "waypoints", 1, std::bind(&OffboardControl::waypoint_callback, this, std::placeholders::_1));

    // Service clients
    arming_client_ = this->create_client<mavros_msgs::srv::CommandBool>("/mavros/cmd/arming");
    set_mode_client_ = this->create_client<mavros_msgs::srv::SetMode>("/mavros/set_mode");

    // Timers
    timer_offboard_ = this->create_wall_timer(100ms,
        std::bind(&OffboardControl::offboardTimerCallback, this));
    timer_setpoint_ = this->create_wall_timer(20ms,
        std::bind(&OffboardControl::setpointTimerCallback, this));

    // Initialize state
    current_state_.connected = false;
    current_state_.armed = false;
    current_state_.mode = "NONE";
  }

private:
  // Store current trajectory
  std::vector<Eigen::Matrix<double, 3, 6>> current_coefficients;
  std::vector<double> segment_durations;
  Eigen::Vector3d current_pos{-2.0, 0.0, 1.5};
  Eigen::Vector3d end_pos;
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
  bool _is_yaw_enabled = false;

  nav_msgs::msg::Odometry _odom;
  rclcpp::Time _final_time = rclcpp::Time(0);
  rclcpp::Time _start_time = rclcpp::Time::max();
  double current_yaw = 0;
  Eigen::MatrixXd yaw_control_points;
  NonUniformBspline yaw_traj;
  double yaw_interval;

  // MAVROS state
  mavros_msgs::msg::State current_state_;
  rclcpp::TimerBase::SharedPtr timer_offboard_;
  rclcpp::TimerBase::SharedPtr timer_setpoint_;

  // Publishers
  rclcpp::Publisher<mavros_msgs::msg::PositionTarget>::SharedPtr setpoint_raw_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;

  // Subscribers
  rclcpp::Subscription<mavros_msgs::msg::State>::SharedPtr state_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr local_pos_sub_;
  rclcpp::Subscription<custom_interface_gym::msg::DesTrajectory>::SharedPtr trajectory_sub_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr _dest_pts_sub;

  // Service clients
  rclcpp::Client<mavros_msgs::srv::CommandBool>::SharedPtr arming_client_;
  rclcpp::Client<mavros_msgs::srv::SetMode>::SharedPtr set_mode_client_;

  Eigen::Vector3d waypoints_{50, 0, 2.5};  // Store waypoints as Eigen::Vector3d
  int counter_{0};

  void waypoint_callback(const nav_msgs::msg::Path::SharedPtr msg) {
    if (!msg->poses.empty()) {
      const auto& last_pose = msg->poses.back();
      waypoints_ = Eigen::Vector3d(
          last_pose.pose.position.x, 
          last_pose.pose.position.y, 
          last_pose.pose.position.z);
    }
  }

  void stateCallback(const mavros_msgs::msg::State::SharedPtr msg) {
    current_state_ = *msg;
  }

  void localPositionCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    current_pos.x() = msg->pose.position.x;
    current_pos.y() = msg->pose.position.y;
    current_pos.z() = msg->pose.position.z;

    // Get yaw from quaternion
    tf2::Quaternion q(
        msg->pose.orientation.x,
        msg->pose.orientation.y,
        msg->pose.orientation.z,
        msg->pose.orientation.w);
    tf2::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);
    current_yaw = yaw;

    // Publish drone marker
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = this->now();
    marker.ns = "drone";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::ARROW;
    marker.action = visualization_msgs::msg::Marker::ADD;

    marker.pose.position.x = current_pos.x();
    marker.pose.position.y = current_pos.y();
    marker.pose.position.z = current_pos.z();
    marker.pose.orientation = msg->pose.orientation;

    marker.scale.x = 1.0;
    marker.scale.y = 0.1;
    marker.scale.z = 0.1;
    marker.color.a = 1.0;
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;

    marker_pub_->publish(marker);
  }

  void trajectoryCallback(const custom_interface_gym::msg::DesTrajectory::SharedPtr msg)
  {
    switch (msg->action) {
      case custom_interface_gym::msg::DesTrajectory::ACTION_ADD: 
      {
        if (msg->trajectory_id < trajectory_id) {
          return;
        }
        handleAddTrajectory(msg);
        break;
      }
      case custom_interface_gym::msg::DesTrajectory::ACTION_ABORT:
        is_aborted = true;
        has_trajectory = false;
        break;
      default:
        sendHoverSetpoint();
        break;
    }
  }

  void offboardTimerCallback()
  {
    // Send setpoint before switching to offboard
    if (counter_ < 20) {
      sendHoverSetpoint();
      counter_++;
      return;
    }

    // Switch to offboard mode if not already in it
    if (current_state_.mode != "OFFBOARD" && current_state_.connected) {
      auto request = std::make_shared<mavros_msgs::srv::SetMode::Request>();
      request->custom_mode = "OFFBOARD";
      
      auto result_future = set_mode_client_->async_send_request(request,
          [this](rclcpp::Client<mavros_msgs::srv::SetMode>::SharedFuture future) {
            auto result = future.get();
            if (!result->mode_sent) {
              RCLCPP_ERROR(this->get_logger(), "Failed to set offboard mode");
            }
          });
    }
    // Arm if not armed
    else if (!current_state_.armed && current_state_.connected) {
      arm();
    }
  }

  void arm()
  {
    auto request = std::make_shared<mavros_msgs::srv::CommandBool::Request>();
    request->value = true;
    
    auto result_future = arming_client_->async_send_request(request,
        [this](rclcpp::Client<mavros_msgs::srv::CommandBool>::SharedFuture future) {
          auto result = future.get();
          if (!result->success) {
            RCLCPP_ERROR(this->get_logger(), "Failed to arm");
          }
        });
  }

  void setpointTimerCallback()
  {
    if (!has_trajectory || is_aborted) {
      if (current_state_.armed) sendHoverSetpoint();
      return;
    }

    double elapsed = (this->get_clock()->now() - _start_time).seconds();
    Eigen::Vector3d des_pos = _traj.getPos(elapsed);
    Eigen::Vector3d des_vel = _traj.getVel(elapsed);
    Eigen::Vector3d des_Acc = _traj.getAcc(elapsed);
    Eigen::Vector3d des_jerk = _traj.getJer(elapsed);

    if(elapsed > _traj.getTotalDuration())
    {
        des_pos = _traj.getPos(_traj.getTotalDuration());
        des_vel = _traj.getVel(_traj.getTotalDuration());
        des_Acc = _traj.getAcc(_traj.getTotalDuration());
        des_jerk = _traj.getJer(_traj.getTotalDuration());
    }

    double distance_to_target = (current_pos - waypoints_).norm();
    if(distance_to_target < 1)
    {
        std::cout << "Close to goal" << std::endl;
        mavros_msgs::msg::PositionTarget sp{};
        sp.header.stamp = this->now();
        sp.coordinate_frame = mavros_msgs::msg::PositionTarget::FRAME_LOCAL_NED;
        sp.type_mask = 
            mavros_msgs::msg::PositionTarget::IGNORE_VX |
            mavros_msgs::msg::PositionTarget::IGNORE_VY |
            mavros_msgs::msg::PositionTarget::IGNORE_VZ |
            mavros_msgs::msg::PositionTarget::IGNORE_AFX |
            mavros_msgs::msg::PositionTarget::IGNORE_AFY |
            mavros_msgs::msg::PositionTarget::IGNORE_AFZ;
        
        sp.position.x = waypoints_.x();
        sp.position.y = waypoints_.y();
        sp.position.z = waypoints_.z();

        double yaw_des = 0.0;
        if(_is_yaw_enabled)
        {
            Eigen::VectorXd yaw_vec = yaw_traj.evaluateDeBoorT(elapsed);
            yaw_des = yaw_vec(0);
        }
        sp.yaw = yaw_des;
        std::cout<<"current yaw: "<<current_yaw<<" desired yaw: "<<sp.yaw<<std::endl;
        std::cout << "Position : " << sp.position.x << ' ' << sp.position.y << ' ' << sp.position.z << std::endl;
        setpoint_raw_pub_->publish(sp);
        return;
    }
    
    Eigen::VectorXd yaw_vec = yaw_traj.evaluateDeBoorT(elapsed);
    double des_yaw = yaw_vec(0);
    
    publishSetpoint(des_pos, des_vel, des_Acc, des_yaw);
  }

  void handleAddTrajectory(const custom_interface_gym::msg::DesTrajectory::SharedPtr msg)
  {
      if(msg->trajectory_id < trajectory_id)
      {
          std::cout << "backward trajectory invalid" << std::endl;
          return;
      }
      has_trajectory = true;
      is_aborted = false;
      hover_command_sent = false;
      _traj.clear();
      trajectory_id = msg->trajectory_id;
      _start_time = msg->header.stamp;
      _final_time = _start_time;
      segment_durations.clear();
      current_coefficients.clear();
      yaw_control_points.resize(0, 0);
      segment_durations = msg->duration_vector;
      num_segments = msg->num_segment;
      order = msg->num_order;

      std::vector<double> array_msg_traj = msg->matrices_flat;
      for (int i = 0; i < segment_durations.size(); ++i) 
      {
          Eigen::Map<const Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> matrix(
              array_msg_traj.data() + i * 3 * 6);
          current_coefficients.push_back(matrix);

          std::chrono::duration<double> duration_in_sec(segment_durations[i]);
          rclcpp::Duration rcl_duration(duration_in_sec);
          _final_time += rcl_duration;
      }
      _traj.setParameters(segment_durations, current_coefficients);

      if(msg->yaw_enabled == custom_interface_gym::msg::DesTrajectory::YAW_ENABLED_TRUE)
      {
          std::cout<<"[add callback debug] yaw reference received: "<<std::endl;
          _is_yaw_enabled = true;
          yaw_control_points = Eigen::MatrixXd::Zero(msg->yaw_control_points.size(), 1);

          for(int i = 0; i<msg->yaw_control_points.size(); i++)
          {
              yaw_control_points(i,0) = msg->yaw_control_points[i];
          }
          yaw_interval = msg->yaw_interval;
          yaw_traj.setUniformBspline(yaw_control_points, 1, yaw_interval);
      }
      std::cout<<"in handle add trajectory callback, traj set successfully"<<std::endl;
  }

  void publishSetpoint(const Eigen::Vector3d& pos, const Eigen::Vector3d& vel,
                         const Eigen::Vector3d& acc, const double &yaw_des)
  {
    mavros_msgs::msg::PositionTarget sp{};
    sp.header.stamp = this->now();
    sp.coordinate_frame = mavros_msgs::msg::PositionTarget::FRAME_LOCAL_NED;
    
    // Position control (ignore velocity and acceleration)
    sp.type_mask = 
        mavros_msgs::msg::PositionTarget::IGNORE_VX |
        mavros_msgs::msg::PositionTarget::IGNORE_VY |
        mavros_msgs::msg::PositionTarget::IGNORE_VZ |
        mavros_msgs::msg::PositionTarget::IGNORE_AFX |
        mavros_msgs::msg::PositionTarget::IGNORE_AFY |
        mavros_msgs::msg::PositionTarget::IGNORE_AFZ;
    
    sp.position.x = pos.x();
    sp.position.y = pos.y();
    sp.position.z = pos.z();
    sp.yaw = yaw_des;
    
    std::cout<<"current yaw: "<<current_yaw<<" desired yaw: "<<sp.yaw<<std::endl;
    setpoint_raw_pub_->publish(sp);
  }

  void sendHoverSetpoint()
  {
    auto dir = waypoints_ - current_pos;

    mavros_msgs::msg::PositionTarget sp{};
    sp.header.stamp = this->now();
    sp.coordinate_frame = mavros_msgs::msg::PositionTarget::FRAME_LOCAL_NED;
    sp.type_mask = 
        mavros_msgs::msg::PositionTarget::IGNORE_VX |
        mavros_msgs::msg::PositionTarget::IGNORE_VY |
        mavros_msgs::msg::PositionTarget::IGNORE_VZ |
        mavros_msgs::msg::PositionTarget::IGNORE_AFX |
        mavros_msgs::msg::PositionTarget::IGNORE_AFY |
        mavros_msgs::msg::PositionTarget::IGNORE_AFZ;
    
    sp.position.x = current_pos.x();
    sp.position.y = current_pos.y();
    sp.position.z = 2.5;
    

    std::cout << "Current Position : " << current_pos.x() << ' ' 
              << current_pos.y() << ' ' << current_pos.z() << std::endl;

    sp.yaw = wrapAngle(atan2(dir.y(), dir.x()));
    setpoint_raw_pub_->publish(sp);
  }

  static double wrapAngle(double angle)
  {
    angle = fmod(angle + M_PI, 2*M_PI);
    return angle >= 0 ? (angle - M_PI) : (angle + M_PI);
  }
};

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<OffboardControl>());
  rclcpp::shutdown();
  return 0;
}