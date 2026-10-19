#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include "propup_ltv_mpc/LTV_MPC.hpp"
#include "propup_ltv_mpc/PredictionModel.hpp"
#include "propup_ltv_mpc/MPCCost.hpp"
#include "propup_ltv_mpc/MPCConstraints.hpp"
#include "propup_ltv_mpc/QPSolver.hpp"

#include <Eigen/Dense>

#include <array>
#include <vector>
#include <chrono>
#include <functional>
#include <cmath>
#include <algorithm>
#include <exception>

using namespace std::chrono_literals;

class MPCSolverNode : public rclcpp::Node
{
public:
    MPCSolverNode()
        : Node("mpc_solver_node"),
          prediction_(HORIZON),
          constraints_(HORIZON, MU, FZ_MAX)
    {
        Eigen::Matrix3d inertia = Eigen::Matrix3d::Zero();
        inertia.diagonal() << 0.1, 0.25, 0.3;

        dynamics_.setRobotParameters(MASS, inertia);

        // State weights
        Eigen::Matrix<double, 13, 13> Q =
            Eigen::Matrix<double, 13, 13>::Identity();

        Q.diagonal() <<
            50.0, 50.0, 50.0,    // Orientation
            10.0, 10.0, 1000.0,  // Position
            1.0, 1.0, 1.0,       // Angular velocity
            1.0, 1.0, 10.0,      // Linear velocity
            0.0;                 // Gravity state

        Eigen::Matrix<double, 12, 12> R =
            Eigen::Matrix<double, 12, 12>::Identity() * 0.001;

        Qbar_ = Eigen::MatrixXd::Zero(
            13 * HORIZON, 13 * HORIZON
        );

        Rbar_ = Eigen::MatrixXd::Zero(
            12 * HORIZON, 12 * HORIZON
        );

        for (int k = 0; k < HORIZON; ++k)
        {
            Qbar_.block<13, 13>(13 * k, 13 * k) = Q;
            Rbar_.block<12, 12>(12 * k, 12 * k) = R;
        }

        cost_.setWeights(Qbar_, Rbar_);

        state_sub_ = create_subscription<
            std_msgs::msg::Float64MultiArray>(
            "/robot_state_estimated",
            10,
            std::bind(
                &MPCSolverNode::state_callback,
                this,
                std::placeholders::_1
            )
        );

        gait_sub_ = create_subscription<
            std_msgs::msg::Float64MultiArray>(
            "/leg_stance_state",
            10,
            std::bind(
                &MPCSolverNode::gait_callback,
                this,
                std::placeholders::_1
            )
        );

        feet_sub_ = create_subscription<
            std_msgs::msg::Float64MultiArray>(
            "/foot_positions",
            10,
            std::bind(
                &MPCSolverNode::feet_callback,
                this,
                std::placeholders::_1
            )
        );

        force_pub_ = create_publisher<
            std_msgs::msg::Float64MultiArray>(
            "/mpc_forces", 10
        );

        timer_ = create_wall_timer(
            20ms,
            std::bind(&MPCSolverNode::control_loop, this)
        );

        RCLCPP_INFO(
            get_logger(),
            "LTV-MPC solver node started in simulation-only mode"
        );
    }

private:
    static constexpr int HORIZON = 10;

    static constexpr double DT = 0.02;
    static constexpr double MASS = 12.0;
    static constexpr double MU = 0.5;
    static constexpr double FZ_MAX = 100.0;

    static constexpr double INPUT_TIMEOUT = 0.2;

    LTV_MPC dynamics_;
    PredictionModel prediction_;
    MPCCost cost_;
    MPCConstraints constraints_;
    QPSolver solver_;

    Eigen::MatrixXd Qbar_;
    Eigen::MatrixXd Rbar_;

    LTV_MPC::StateVector robot_state_ =
        LTV_MPC::StateVector::Zero();

    std::array<int, 4> stance_state_{};

    std::array<Eigen::Vector3d, 4> feet_{};

    bool state_received_ = false;
    bool gait_received_ = false;
    bool feet_received_ = false;

    rclcpp::Time state_time_;
    rclcpp::Time gait_time_;
    rclcpp::Time feet_time_;

    rclcpp::Subscription<
        std_msgs::msg::Float64MultiArray>::SharedPtr state_sub_;

    rclcpp::Subscription<
        std_msgs::msg::Float64MultiArray>::SharedPtr gait_sub_;

    rclcpp::Subscription<
        std_msgs::msg::Float64MultiArray>::SharedPtr feet_sub_;

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
            state_received_ = false;
            return;
        }

        for (int i = 0; i < 13; ++i)
        {
            if (!std::isfinite(msg->data[i]))
            {
                state_received_ = false;
                return;
            }

            robot_state_(i) = msg->data[i];
        }

        // The 13th state represents gravity.
        // Enforce the expected world-frame convention.
        if (std::abs(robot_state_(12) + 9.81) > 0.1)
        {
            state_received_ = false;
            RCLCPP_WARN(
                get_logger(),
                "Invalid gravity state"
            );
            return;
        }

        state_time_ = now();
        state_received_ = true;
    }

    void gait_callback(
        const std_msgs::msg::Float64MultiArray::SharedPtr msg)
    {
        if (msg->data.size() != 4)
        {
            gait_received_ = false;
            return;
        }

        for (int i = 0; i < 4; ++i)
        {
            if (!std::isfinite(msg->data[i]))
            {
                gait_received_ = false;
                return;
            }

            if (msg->data[i] != 0.0 &&
                msg->data[i] != 1.0)
            {
                gait_received_ = false;
                return;
            }

            stance_state_[i] =
                static_cast<int>(msg->data[i]);
        }

        gait_time_ = now();
        gait_received_ = true;
    }

    void feet_callback(
        const std_msgs::msg::Float64MultiArray::SharedPtr msg)
    {
        if (msg->data.size() != 12)
        {
            feet_received_ = false;
            return;
        }

        for (int leg = 0; leg < 4; ++leg)
        {
            for (int axis = 0; axis < 3; ++axis)
            {
                const double value =
                    msg->data[leg * 3 + axis];

                if (!std::isfinite(value))
                {
                    feet_received_ = false;
                    return;
                }

                feet_[leg](axis) = value;
            }
        }

        feet_time_ = now();
        feet_received_ = true;
    }

    bool inputs_fresh()
    {
        if (!state_received_ ||
            !gait_received_ ||
            !feet_received_)
        {
            return false;
        }

        const auto current_time = now();

        const double state_age =
            (current_time - state_time_).seconds();

        const double gait_age =
            (current_time - gait_time_).seconds();

        const double feet_age =
            (current_time - feet_time_).seconds();

        return state_age >= 0.0 &&
               gait_age >= 0.0 &&
               feet_age >= 0.0 &&
               state_age < INPUT_TIMEOUT &&
               gait_age < INPUT_TIMEOUT &&
               feet_age < INPUT_TIMEOUT;
    }

    void publish_zero_forces()
    {
        std_msgs::msg::Float64MultiArray msg;
        msg.data.assign(12, 0.0);
        force_pub_->publish(msg);
    }

    void control_loop()
{
    auto start_time = std::chrono::steady_clock::now();

    if (!inputs_fresh())
        {
            publish_zero_forces();
            return;
        }

        try
        {
            // =====================================
            // 1. Generate dynamics sequence
            // =====================================

            // =====================================
// 1. Generate time-varying dynamics
// =====================================

std::vector<LTV_MPC::StateMatrix> A_sequence;
std::vector<LTV_MPC::InputMatrix> B_sequence;

A_sequence.reserve(HORIZON);
B_sequence.reserve(HORIZON);

// Begin prediction from the measured robot state
LTV_MPC::StateVector predicted_state = robot_state_;

// Construct nominal ground reaction forces
Eigen::Matrix<double, 12, 1> nominal_input =
    Eigen::Matrix<double, 12, 1>::Zero();

int supporting_legs = 0;

for (int leg = 0; leg < 4; ++leg)
{
    if (stance_state_[leg] == 1)
    {
        ++supporting_legs;
    }
}

if (supporting_legs > 0)
{
    const double nominal_force =
        MASS * 9.81 / supporting_legs;

    for (int leg = 0; leg < 4; ++leg)
    {
        if (stance_state_[leg] == 1)
        {
            nominal_input(3 * leg + 2) =
                nominal_force;
        }
    }
}

// Predict dynamics along the nominal trajectory
for (int k = 0; k < HORIZON; ++k)
{
    dynamics_.computeContinuousDynamics(
        predicted_state,
        feet_
    );

    dynamics_.discretizeDynamics(DT);

    const auto Ad = dynamics_.getAd();
    const auto Bd = dynamics_.getBd();

    A_sequence.push_back(Ad);
    B_sequence.push_back(Bd);

    // Propagate the predicted robot state
    predicted_state =
        Ad * predicted_state +
        Bd * nominal_input;
}

            // =====================================
            // 2. Prediction matrices
            // =====================================

            auto prediction_start = std::chrono::steady_clock::now();

prediction_.computeTimeVaryingPredictionMatrices(
    A_sequence,
    B_sequence
);

auto prediction_end = std::chrono::steady_clock::now();

double prediction_time_ms =
    std::chrono::duration<double, std::milli>(
        prediction_end - prediction_start
    ).count();

RCLCPP_INFO_THROTTLE(
    get_logger(),
    *get_clock(),
    1000,
    "Prediction matrix time: %.3f ms",
    prediction_time_ms
);

            // =====================================
            // 3. Standing reference trajectory
            // =====================================

            Eigen::VectorXd X_ref =
                Eigen::VectorXd::Zero(13 * HORIZON);

            LTV_MPC::StateVector reference =
                LTV_MPC::StateVector::Zero();

            reference(3) = robot_state_(3);
            reference(4) = robot_state_(4);
            reference(5) = 0.30;
            reference(12) = -9.81;

            for (int k = 0; k < HORIZON; ++k)
            {
                X_ref.segment<13>(13 * k) = reference;
            }

            // =====================================
            // 4. MPC cost
            // =====================================

            auto cost_start = std::chrono::steady_clock::now();

// ==========================================
// Generate nominal ground reaction forces
// ==========================================

Eigen::VectorXd U_ref =
    Eigen::VectorXd::Zero(12 * HORIZON);

// Count supporting legs
int stance_count = 0;

for (int leg = 0; leg < 4; ++leg)
{
    if (stance_state_[leg] == 1)
    {
        ++stance_count;
    }
}

// Calculate nominal vertical force
double nominal_fz = 0.0;

if (stance_count > 0)
{
    nominal_fz = MASS * 9.81 / stance_count;
}

// Fill nominal force vector across the horizon
for (int k = 0; k < HORIZON; ++k)
{
    for (int leg = 0; leg < 4; ++leg)
    {
        if (stance_state_[leg] == 1)
        {
            U_ref(k * 12 + leg * 3 + 2) =
                nominal_fz;
        }
    }
}

// ==========================================
// Compute MPC cost with nominal forces
// ==========================================

cost_.computeCost(
    prediction_.getP(),
    prediction_.getH(),
    robot_state_,
    X_ref,
    U_ref
);

auto cost_end = std::chrono::steady_clock::now();

double cost_time_ms =
    std::chrono::duration<double, std::milli>(
        cost_end - cost_start
    ).count();

RCLCPP_INFO_THROTTLE(
    get_logger(),
    *get_clock(),
    1000,
    "MPC cost time: %.3f ms",
    cost_time_ms
);

            // =====================================
            // 5. Contact constraints
            // =====================================

            constraints_.buildConstraints(stance_state_);

            // =====================================
            // 6. Solve QP
            // =====================================

           // Start measuring QP solver time
auto qp_start = std::chrono::steady_clock::now();

const bool success = solver_.solve(
    cost_.getHessian(),
    cost_.getGradient(),
    constraints_.getC(),
    constraints_.getLowerBound(),
    constraints_.getUpperBound()
);

// Stop measuring QP solver time
auto qp_end = std::chrono::steady_clock::now();

double qp_time_ms =
    std::chrono::duration<double, std::milli>(
        qp_end - qp_start
    ).count();

RCLCPP_INFO_THROTTLE(
    get_logger(),
    *get_clock(),
    1000,
    "OSQP solver time: %.3f ms",
    qp_time_ms
);

            if (!success)
            {
                RCLCPP_WARN(
                    get_logger(),
                    "MPC optimization failed"
                );

                publish_zero_forces();
                return;
            }

            const Eigen::VectorXd& solution =
                solver_.getSolution();

            if (solution.size() < 12 ||
                !solution.allFinite())
            {
                publish_zero_forces();
                return;
            }

            // =====================================
            // 7. Publish first optimized input
            // =====================================

            std_msgs::msg::Float64MultiArray forces;
            forces.data.resize(12);

            for (int i = 0; i < 12; ++i)
            {
                forces.data[i] = solution(i);
            }

            force_pub_->publish(forces);
auto end_time = std::chrono::steady_clock::now();

double elapsed_ms =
    std::chrono::duration<double, std::milli>(
        end_time - start_time
    ).count();

RCLCPP_INFO_THROTTLE(
    get_logger(),
    *get_clock(),
    1000,
    "MPC computation time: %.3f ms",
    elapsed_ms
);
        }
        catch (const std::exception& e)
        {
            RCLCPP_ERROR(
                get_logger(),
                "MPC exception: %s",
                e.what()
            );

            publish_zero_forces();
        }
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    rclcpp::spin(
        std::make_shared<MPCSolverNode>()
    );

    rclcpp::shutdown();

    return 0;
}
