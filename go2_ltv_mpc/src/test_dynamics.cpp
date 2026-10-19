#include <iostream>
#include <Eigen/Dense>
#include <array>

#include "propup_ltv_mpc/LTV_MPC.hpp"

int main()
{
    // Create MPC dynamics object
    LTV_MPC mpc;

    // Robot mass (kg)
    double mass = 12.0;

    // Robot inertia matrix (kg m^2)
    Eigen::Matrix3d inertia = Eigen::Matrix3d::Identity();

    // Set robot parameters
    mpc.setRobotParameters(mass, inertia);

    // 13-dimensional robot state
    LTV_MPC::StateVector state =
        LTV_MPC::StateVector::Zero();

    // Four foot positions relative to robot COM
    std::array<Eigen::Vector3d, 4> foot_positions = {
        Eigen::Vector3d( 0.2,  0.1, -0.3),
        Eigen::Vector3d( 0.2, -0.1, -0.3),
        Eigen::Vector3d(-0.2,  0.1, -0.3),
        Eigen::Vector3d(-0.2, -0.1, -0.3)
    };

    // Compute continuous-time dynamics
    mpc.computeContinuousDynamics(state, foot_positions);

    // Discretization time step
    double dt = 0.02;

    // Compute discrete-time dynamics
    mpc.discretizeDynamics(dt);

    // Get matrices
    auto Ac = mpc.getA();
    auto Bc = mpc.getB();

    auto Ad = mpc.getAd();
    auto Bd = mpc.getBd();

    // Display dimensions
    std::cout << "Ac dimensions: "
              << Ac.rows() << " x " << Ac.cols()
              << std::endl;

    std::cout << "Bc dimensions: "
              << Bc.rows() << " x " << Bc.cols()
              << std::endl;

    std::cout << "Ad dimensions: "
              << Ad.rows() << " x " << Ad.cols()
              << std::endl;

    std::cout << "Bd dimensions: "
              << Bd.rows() << " x " << Bd.cols()
              << std::endl;

    // Verify Euler discretization
    double error_A =
        (Ad - (LTV_MPC::StateMatrix::Identity()
               + Ac * dt)).norm();

    double error_B =
        (Bd - Bc * dt).norm();

    std::cout << "\nDiscretization error A: "
              << error_A << std::endl;

    std::cout << "Discretization error B: "
              << error_B << std::endl;

    return 0;
}
