#include <rclcpp/rclcpp.hpp>
#include <memory>
#include <string>
#include <rog_map_ros/rog_map_ros2.hpp>

class ROGMapTestNode : public rclcpp::Node
{
public:
    ROGMapTestNode()
        : Node("rog_map_test_node")
    {
        // Declare parameter
        // this->declare_parameter<std::string>("config_file", "config/rog_map.yaml");
        this->declare_parameter<std::string>("config_file", "/home/astik/super_ws/src/SUPER/rog_map/config/rog_map.yaml");
    }

    void init()
    {

        std::string config_file = this->get_parameter("config_file").as_string();
        RCLCPP_INFO(this->get_logger(), "Using config file: %s", config_file.c_str());

        // Disambiguate shared_from_this to use rclcpp::Node's version
        rog_map_ros_ = std::make_shared<rog_map::ROGMapROS>(this->rclcpp::Node::shared_from_this(), config_file);

        if (!rog_map_ros_) {
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize ROGMapROS!");
            rclcpp::shutdown();
        } else {
            RCLCPP_INFO(this->get_logger(), "ROGMapROS initialized successfully.");
        }
    }

private:
    std::shared_ptr<rog_map::ROGMapROS> rog_map_ros_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<ROGMapTestNode>();
    node->init();

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
