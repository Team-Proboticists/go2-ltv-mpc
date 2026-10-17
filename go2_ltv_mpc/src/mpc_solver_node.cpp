#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include <array>
#include <chrono>
#include <functional>

using namespace std::chrono_literals;

class MPCSolverNode : public rclcpp::Node
{
public:
    MPCSolverNode() : Node("mpc_solver_node")
    {
        // Receive the 13-state vector from Module 1
        state_sub_ = create_subscription<
            std_msgs::msg::Float64MultiArray>(
            "/robot_state_estimated",
            10,
            std::bind(
                &MPCSolverNode::state_callback,
                this,
                std::placeholders::_1)
        );

        // Receive stance/swing information
        gait_sub_ = create_subscription<
            std_msgs::msg::Float64MultiArray>(
            "/leg_stance_state",
            10,
            std::bind(
                &MPCSolverNode::gait_callback,
                this,
                std::placeholders::_1)
        );

        // Publish optimized forces
        force_pub_ = create_publisher<
            std_msgs::msg::Float64MultiArray>(
            "/mpc_forces", 10
        );

        // Run at 50 Hz
        timer_ = create_wall_timer(
            20ms,
            std::bind(&MPCSolverNode::control_loop, this)
        );

        RCLCPP_INFO(
            get_logger(),
            "MPC Solver Node Started"
        );
    }

private:

    std::array<double, 13> robot_state_{};
    std::array<double, 4> stance_state_{};

    bool state_received_ = false;
    bool gait_received_ = false;

    rclcpp::Subscription<
        std_msgs::msg::Float64MultiArray>::SharedPtr state_sub_;

    rclcpp::Subscription<
        std_msgs::msg::Float64MultiArray>::SharedPtr gait_sub_;

    rclcpp::Publisher<
        std_msgs::msg::Float64MultiArray>::SharedPtr force_pub_;

    rclcpp::TimerBase::SharedPtr timer_;

    void state_callback(
        const std_msgs::msg::Float64MultiArray::SharedPtr msg)
    {
        if (msg->data.size() != 13)
        {
            RCLCPP_WARN(
                get_logger(),
                "Expected 13 state values"
            );
            return;
        }

        for (size_t i = 0; i < 13; i++)
            robot_state_[i] = msg->data[i];

        state_received_ = true;
    }

    void gait_callback(
        const std_msgs::msg::Float64MultiArray::SharedPtr msg)
    {
        if (msg->data.size() != 4)
        {
            RCLCPP_WARN(
                get_logger(),
                "Expected 4 gait values"
            );
            return;
        }

        for (size_t i = 0; i < 4; i++)
            stance_state_[i] = msg->data[i];

        gait_received_ = true;
    }

    void control_loop()
    {
        if (!state_received_ || !gait_received_)
            return;

        std_msgs::msg::Float64MultiArray forces;

        // Temporary dummy forces.
        // This is NOT MPC optimization.
        forces.data.resize(12, 0.0);

        // Example: each stance leg gets 20 N upward.
        for (size_t leg = 0; leg < 4; leg++)
        {
            if (stance_state_[leg] > 0.5)
            {
                forces.data[leg * 3 + 2] = 20.0;
            }
        }

        force_pub_->publish(forces);
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    rclcpp::spin(
        std::make_shared<MPCSolverNode>()
    );

    rclcpp::shutdown();
    return 0;
}
