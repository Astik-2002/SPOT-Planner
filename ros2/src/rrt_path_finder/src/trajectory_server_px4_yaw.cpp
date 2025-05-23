#include <rclcpp/rclcpp.hpp>
#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
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
    // Publishers
    offboard_control_mode_pub_ = this->create_publisher<px4_msgs::msg::OffboardControlMode>(
        "/fmu/in/offboard_control_mode", 10);
    trajectory_setpoint_pub_ = this->create_publisher<px4_msgs::msg::TrajectorySetpoint>(
        "/fmu/in/trajectory_setpoint", 10);
    vehicle_command_pub_ = this->create_publisher<px4_msgs::msg::VehicleCommand>(
        "/fmu/in/vehicle_command", 10);
    marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
        "/drone_marker", 10);

    // Subscribers
    vehicle_odometry_sub_ = this->create_subscription<px4_msgs::msg::VehicleOdometry>(
        "/fmu/out/vehicle_odometry", rclcpp::QoS(10).best_effort(),
        std::bind(&OffboardControl::odometryCallback, this, std::placeholders::_1));
    trajectory_sub_ = this->create_subscription<custom_interface_gym::msg::DesTrajectory>(
        "/des_trajectory", 1,
        std::bind(&OffboardControl::trajectoryCallback, this, std::placeholders::_1));

    _dest_pts_sub = this->create_subscription<nav_msgs::msg::Path>(
            "waypoints", 1,  std::bind(&OffboardControl::waypoint_callback, this, std::placeholders::_1));

    // Timers
    timer_offboard_ = this->create_wall_timer(100ms,
        std::bind(&OffboardControl::offboardTimerCallback, this));
    timer_setpoint_ = this->create_wall_timer(20ms,  // Increased rate for smoother control
        std::bind(&OffboardControl::setpointTimerCallback, this));
  }

private:
  // Store current trajectory
  std::vector<Eigen::Matrix<double, 3, 6>> current_coefficients;  // For each segment
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
  // Publishers
  rclcpp::Publisher<px4_msgs::msg::OffboardControlMode>::SharedPtr offboard_control_mode_pub_;
  rclcpp::Publisher<px4_msgs::msg::TrajectorySetpoint>::SharedPtr trajectory_setpoint_pub_;
  rclcpp::Publisher<px4_msgs::msg::VehicleCommand>::SharedPtr vehicle_command_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;

  // Subscribers
  rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr vehicle_odometry_sub_;
  rclcpp::Subscription<custom_interface_gym::msg::DesTrajectory>::SharedPtr trajectory_sub_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr _dest_pts_sub;

  Eigen::Vector3d waypoints_;  // Store waypoints as Eigen::Vector3d
  Eigen::Vector3d waypoints_NED;  // Store waypoints as Eigen::Vector3d

  // Timers
  rclcpp::TimerBase::SharedPtr timer_offboard_;
  rclcpp::TimerBase::SharedPtr timer_setpoint_;

  // State variables
  uint32_t current_trajectory_id_{0};
  bool has_trajectory_{false};
  bool is_aborted_{false};
  std::vector<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> current_coefficients_;
  std::vector<double> segment_durations_;
  rclcpp::Time start_time_;
  rclcpp::Time final_time_;

  Eigen::Vector3d current_pos_{0, 0, 0};
  int counter_{0};
  bool armed_{false};

  void waypoint_callback(const nav_msgs::msg::Path::SharedPtr msg) {
    if (!msg->poses.empty()) {
      const auto& last_pose = msg->poses.back();
      waypoints_ = Eigen::Vector3d(
          last_pose.pose.position.x, 
          last_pose.pose.position.y, 
          last_pose.pose.position.z);
      
      waypoints_NED = Eigen::Vector3d(
        last_pose.pose.position.y, 
        last_pose.pose.position.x, 
        -last_pose.pose.position.z);
    }
  }

  void trajectoryCallback(const custom_interface_gym::msg::DesTrajectory::SharedPtr msg)
  {
    switch (msg->action) {
      case custom_interface_gym::msg::DesTrajectory::ACTION_ADD: 
      {
        if (msg->trajectory_id < current_trajectory_id_) {
          return;
        }
        handleAddTrajectory(msg);
        break;
      }
      case custom_interface_gym::msg::DesTrajectory::ACTION_ABORT:
        is_aborted_ = true;
        has_trajectory_ = false;
        break;
      default:
        sendHoverSetpoint();
        break;
    }
  }

  void handleFinalTrajectory()
  {
      sendHoverSetpoint();
  }

  void odometryCallback(const px4_msgs::msg::VehicleOdometry::SharedPtr msg)
  {
    // Convert NED to ENU
    current_pos_.x() = msg->position[0];  // East (Y in NED)
    current_pos_.y() = -msg->position[1];   // North (X in NED)
    current_pos_.z() = -msg->position[2];   // Up (-Z in NED)

    // Publish drone marker
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "map"; // Match ENU frame
    marker.header.stamp = this->now();
    marker.ns = "drone";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::ARROW;
    marker.action = visualization_msgs::msg::Marker::ADD;

    marker.pose.position.x = current_pos_.x();
    marker.pose.position.y = current_pos_.y();
    marker.pose.position.z = current_pos_.z();

    // Convert orientation from NED to ENU
    tf2::Quaternion q_ned(msg->q[1], msg->q[2], msg->q[3], msg->q[0]);
    tf2::Quaternion q_enu(msg->q[1], -msg->q[2], -msg->q[3], msg->q[0]);
    double roll, pitch, yaw;
    tf2::Matrix3x3(q_enu).getRPY(roll, pitch, yaw);
    current_yaw = yaw;
    marker.pose.orientation.x = q_ned.x();
    marker.pose.orientation.y = q_ned.y();
    marker.pose.orientation.z = q_ned.z();
    marker.pose.orientation.w = q_ned.w();

    marker.scale.x = 1.0;
    marker.scale.y = 0.1;
    marker.scale.z = 0.1;
    marker.color.a = 1.0;
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;

    marker_pub_->publish(marker);
  }

  void offboardTimerCallback()
  {
    px4_msgs::msg::OffboardControlMode offb_msg;
    offb_msg.position = true;
    offb_msg.timestamp = this->now().nanoseconds() / 1000;
    offboard_control_mode_pub_->publish(offb_msg);

    if (++counter_ == 20 && !armed_) {
      arm();
      setOffboardMode();
    }
  }

  void arm()
  {
    auto cmd = px4_msgs::msg::VehicleCommand();
    cmd.command = px4_msgs::msg::VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM;
    cmd.param1 = 1.0;
    cmd.target_system = 1;
    cmd.timestamp = this->now().nanoseconds() / 1000;
    vehicle_command_pub_->publish(cmd);
    armed_ = true;
  }

  void setOffboardMode()
  {
    auto cmd = px4_msgs::msg::VehicleCommand();
    cmd.command = px4_msgs::msg::VehicleCommand::VEHICLE_CMD_DO_SET_MODE;
    cmd.param1 = 1;
    cmd.param2 = 6; // Offboard mode
    cmd.target_system = 1;
    cmd.timestamp = this->now().nanoseconds() / 1000;
    vehicle_command_pub_->publish(cmd);
  }

  void setpointTimerCallback()
  {
    if (!has_trajectory_ || is_aborted_) {
        if (armed_) sendHoverSetpoint();
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

    double distance_to_target = (current_pos_ - waypoints_).norm();
    if(distance_to_target < 1)
    {
        std::cout << "Close to goal" << std::endl;
        px4_msgs::msg::TrajectorySetpoint sp{};
        sp.timestamp = this->now().nanoseconds() / 1000;
        sp.position[0] = waypoints_[0];
        sp.position[1] = -waypoints_[1];
        sp.position[2] = -waypoints_[2];

        sp.velocity[0] = std::numeric_limits<float>::quiet_NaN();
        sp.velocity[1] = std::numeric_limits<float>::quiet_NaN();
        sp.velocity[2] = std::numeric_limits<float>::quiet_NaN();

        sp.acceleration[0] = std::numeric_limits<float>::quiet_NaN();
        sp.acceleration[1] = std::numeric_limits<float>::quiet_NaN();
        sp.acceleration[2] = std::numeric_limits<float>::quiet_NaN();
        double yaw_des = 0.0;
        if(_is_yaw_enabled)
        {
            Eigen::VectorXd yaw_vec = yaw_traj.evaluateDeBoorT(elapsed);
            yaw_des = yaw_vec(0);
        }
        sp.yaw = yaw_des; // wrapAngle(yaw_ned);
        std::cout<<"current yaw: "<<current_yaw<<" desired yaw: "<<sp.yaw<<std::endl;
        std::cout << "Position : " << sp.position[0] << ' ' << sp.position[1] << ' ' << sp.position[2] << std::endl;
        trajectory_setpoint_pub_->publish(sp);
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
      has_trajectory_ = true;
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

  
  void publishSetpoint(const Eigen::Vector3d& pos_enu, const Eigen::Vector3d& vel_enu,
                         const Eigen::Vector3d& acc_enu, const double &yaw_des)
  {
    px4_msgs::msg::TrajectorySetpoint sp{};
    sp.timestamp = this->now().nanoseconds() / 1000;

    // Convert ENU to NED (PX4 coordinate system)
    sp.position[0] = pos_enu.x();   // North (ENU Y)
    sp.position[1] = -pos_enu.y();   // East (ENU X)
    sp.position[2] = -pos_enu.z();   // Down (ENU -Z)

    sp.velocity[0] = vel_enu.x();    // North velocity
    sp.velocity[1] = -vel_enu.y();    // East velocity
    sp.velocity[2] = -vel_enu.z();    // Down velocity

    sp.acceleration[0] = acc_enu.x(); // North acceleration
    sp.acceleration[1] = -acc_enu.y(); // East acceleration
    sp.acceleration[2] = -acc_enu.z(); // Down acceleration
    Eigen::Vector3d des_pos_NED{sp.position[0], sp.position[1], sp.position[2]};
    sp.yaw = -yaw_des; // wrapAngle(yaw_ned);
    std::cout<<"current yaw: "<<current_yaw<<" desired yaw: "<<sp.yaw<<std::endl;
    trajectory_setpoint_pub_->publish(sp);
  }

  void sendHoverSetpoint()
  {
    auto dir = waypoints_ - current_pos;

    px4_msgs::msg::TrajectorySetpoint sp{};
    sp.timestamp = this->now().nanoseconds() / 1000;
    sp.position[0] = current_pos_.x();
    sp.position[1] = -current_pos_.y();
    sp.position[2] = -3.0;

    sp.velocity[0] = std::numeric_limits<float>::quiet_NaN();
    sp.velocity[1] = std::numeric_limits<float>::quiet_NaN();
    sp.velocity[2] = std::numeric_limits<float>::quiet_NaN();

    sp.acceleration[0] = std::numeric_limits<float>::quiet_NaN();
    sp.acceleration[1] = std::numeric_limits<float>::quiet_NaN();
    sp.acceleration[2] = std::numeric_limits<float>::quiet_NaN();

    std::cout << "Current Position : " << current_pos_.x() << ' ' 
              << current_pos_.y() << ' ' << current_pos_.z() << std::endl;

    sp.yaw = wrapAngle(atan2(dir.y(), dir.x()) - M_PI);
    trajectory_setpoint_pub_->publish(sp);
  }

  void sendLandCommand()
  {
    auto cmd = px4_msgs::msg::VehicleCommand{};
    cmd.command = px4_msgs::msg::VehicleCommand::VEHICLE_CMD_NAV_LAND;
    cmd.param5 = current_pos_.y();
    cmd.param6 = current_pos_.x();
    cmd.param7 = -current_pos_.z();
    cmd.target_system = 1;
    cmd.timestamp = this->now().nanoseconds() / 1000;
    vehicle_command_pub_->publish(cmd);
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