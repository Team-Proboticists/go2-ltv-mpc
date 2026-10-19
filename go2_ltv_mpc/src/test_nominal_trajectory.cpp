#include "propup_ltv_mpc/LTV_MPC.hpp"

#include <Eigen/Dense>
#include <iostream>
#include <array>
#include <vector>
#include <cmath>

int main()
{
    const int horizon = 10;
    const double dt = 0.02;
    const double mass = 12.0;

    // ==========================================
    // 1. Initialize robot dynamics
    // ==========================================

    LTV_MPC dynamics;

    Eigen::Matrix3d inertia =
        Eigen::Matrix3d::Zero();

    inertia.diagonal() << 0.1, 0.25, 0.3;

    dynamics.setRobotParameters(mass, inertia);

    // ==========================================
    // 2. Initial robot state
    // ==========================================

    LTV_MPC::StateVector x =
        LTV_MPC::StateVector::Zero();

    x(5) = 0.30;
    x(12) = -9.81;

    // ==========================================
    // 3. Foot positions relative to COM
    // ==========================================

    std::array<Eigen::Vector3d, 4> feet;

    feet[0] = Eigen::Vector3d( 0.20,  0.10, -0.30);
    feet[1] = Eigen::Vector3d( 0.20, -0.10, -0.30);
    feet[2] = Eigen::Vector3d(-0.20,  0.10, -0.30);
    feet[3] = Eigen::Vector3d(-0.20, -0.10, -0.30);

    // ==========================================
    // 4. Nominal ground reaction forces
    // ==========================================

    Eigen::Matrix<double, 12, 1> u =
        Eigen::Matrix<double, 12, 1>::Zero();

    const double nominal_fz = mass * 9.81 / 4.0;

    for (int leg = 0; leg < 4; ++leg)
    {
        u(leg * 3 + 2) = nominal_fz;
    }

    std::cout << "Nominal force per foot: "
              << nominal_fz << " N\n\n";

    // ==========================================
    // 5. Store predicted states
    // ==========================================

    std::vector<LTV_MPC::StateVector> trajectory;

    trajectory.reserve(horizon + 1);
    trajectory.push_back(x);

    // ==========================================
    // 6. Generate nominal trajectory
    // ==========================================

    for (int k = 0; k < horizon; ++k)
    {
        dynamics.computeContinuousDynamics(x, feet);
        dynamics.discretizeDynamics(dt);

        const auto Ad = dynamics.getAd();
        const auto Bd = dynamics.getBd();

        x = Ad * x + Bd * u;

        trajectory.push_back(x);

        std::cout
            << "Step " << k + 1
            << " | Height: " << x(5)
            << " | Vertical velocity: " << x(11)
            << " | Yaw: " << x(2)
            << "\n";
    }

    // ==========================================
    // 7. Verify standing equilibrium
    // ==========================================

    const double height_error =
        std::abs(x(5) - 0.30);

    const double velocity_error =
        std::abs(x(11));

    std::cout << "\nFinal height error: "
              << height_error << "\n";

    std::cout << "Final vertical velocity error: "
              << velocity_error << "\n";

    if (height_error > 1e-8 ||
        velocity_error > 1e-8)
    {
        std::cerr
            << "\nSTANDING EQUILIBRIUM TEST FAILED!\n";

        return 1;
    }

    std::cout
        << "\nNOMINAL TRAJECTORY TEST PASSED!\n";

    return 0;
}
