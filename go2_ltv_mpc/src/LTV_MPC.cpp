#include "propup_ltv_mpc/LTV_MPC.hpp"

#include <cmath>
#include <stdexcept>

// Constructor
LTV_MPC::LTV_MPC()
{
    mass_ = 1.0;

    inertia_.setIdentity();

    A_c_.setZero();
    B_c_.setZero();
}

// Set robot mass and inertia
void LTV_MPC::setRobotParameters(
    double mass,
    const Eigen::Matrix3d& inertia)
{
    if (mass <= 0.0) {
        throw std::invalid_argument(
            "Robot mass must be positive");
    }

    if (inertia.determinant() <= 0.0) {
        throw std::invalid_argument(
            "Robot inertia must be positive definite");
    }

    mass_ = mass;
    inertia_ = inertia;
}

// Skew-symmetric matrix
Eigen::Matrix3d LTV_MPC::skew(
    const Eigen::Vector3d& v)
{
    Eigen::Matrix3d S;

    S <<  0.0, -v.z(),  v.y(),
          v.z(), 0.0, -v.x(),
         -v.y(), v.x(), 0.0;

    return S;
}

// Compute continuous-time dynamics
void LTV_MPC::computeContinuousDynamics(
    const StateVector& state,
    const std::array<Eigen::Vector3d, 4>& foot_positions)
{
    A_c_.setZero();
    B_c_.setZero();

    // State:
    // [position, linear_velocity, rpy,
    //  angular_velocity, gravity]

    const double yaw = state(8);

    // Rotation about Z axis
    Eigen::Matrix3d Rz;

    Rz << std::cos(yaw), -std::sin(yaw), 0.0,
          std::sin(yaw),  std::cos(yaw), 0.0,
          0.0,            0.0,           1.0;

    // Orientation dynamics
    A_c_.block<3,3>(6,9) = Rz.transpose();

    // Position dynamics
    A_c_.block<3,3>(0,3).setIdentity();

    // Gravity contribution
    A_c_(5,12) = -1.0;

    // Inertia in world frame
    Eigen::Matrix3d I_world =
        Rz * inertia_ * Rz.transpose();

    Eigen::Matrix3d I_inv =
        I_world.inverse();

    // Each leg contributes 3 forces
    for (int leg = 0; leg < 4; leg++)
    {
        int col = leg * 3;

        // Angular acceleration contribution
        B_c_.block<3,3>(9,col) =
            I_inv * skew(foot_positions[leg]);

        // Linear acceleration contribution
        B_c_.block<3,3>(3,col) =
            Eigen::Matrix3d::Identity() / mass_;
    }
}

// Return A matrix
const LTV_MPC::StateMatrix&
LTV_MPC::getA() const
{
    return A_c_;
}

// Return B matrix
const LTV_MPC::InputMatrix&
LTV_MPC::getB() const
{
    return B_c_;
}
// ==========================================
// Discretize continuous-time dynamics
// ==========================================

void LTV_MPC::discretizeDynamics(double dt)
{
    // Forward Euler discretization
    A_d_ = StateMatrix::Identity() + A_c_ * dt;

    B_d_ = B_c_ * dt;
}


// ==========================================
// Return discrete-time A matrix
// ==========================================

const LTV_MPC::StateMatrix& LTV_MPC::getAd() const
{
    return A_d_;
}


// ==========================================
// Return discrete-time B matrix
// ==========================================

const LTV_MPC::InputMatrix& LTV_MPC::getBd() const
{
    return B_d_;
}
