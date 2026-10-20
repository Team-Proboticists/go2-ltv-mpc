#include "propup_ltv_mpc/LTV_MPC.hpp"
#include "propup_ltv_mpc/PredictionModel.hpp"

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

    using StateMatrix = LTV_MPC::StateMatrix;
    using InputMatrix = LTV_MPC::InputMatrix;
    using StateVector = LTV_MPC::StateVector;

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

    StateVector x = StateVector::Zero();

    x(2) = 0.30;

    // Initial yaw angular velocity
    x(11) = 0.20;

    // Gravity
    x(12) = 9.81;

    // ==========================================
    // 3. Fixed world-frame foot positions
    //    relative to stationary COM
    // ==========================================

    std::array<Eigen::Vector3d, 4> feet;

    feet[0] = Eigen::Vector3d( 0.20,  0.10, -0.30);
    feet[1] = Eigen::Vector3d( 0.20, -0.10, -0.30);
    feet[2] = Eigen::Vector3d(-0.20,  0.10, -0.30);
    feet[3] = Eigen::Vector3d(-0.20, -0.10, -0.30);

    // ==========================================
    // 4. Nominal support forces
    // ==========================================

    Eigen::Matrix<double, 12, 1> u =
        Eigen::Matrix<double, 12, 1>::Zero();

    for (int leg = 0; leg < 4; ++leg)
    {
        u(leg * 3 + 2) = mass * 9.81 / 4.0;
    }

    // ==========================================
    // 5. Store time-varying matrices
    // ==========================================

    std::vector<StateMatrix> A_sequence;
    std::vector<InputMatrix> B_sequence;

    A_sequence.reserve(horizon);
    B_sequence.reserve(horizon);

    // ==========================================
    // 6. Generate nominal trajectory
    // ==========================================

    std::cout << "Generating turning trajectory:\n\n";

    for (int k = 0; k < horizon; ++k)
    {
        dynamics.computeContinuousDynamics(x, feet);
        dynamics.discretizeDynamics(dt);

        StateMatrix Ad = dynamics.getAd();
        InputMatrix Bd = dynamics.getBd();

        A_sequence.push_back(Ad);
        B_sequence.push_back(Bd);

        // Propagate nominal state
        x = Ad * x + Bd * u;

        std::cout
            << "Step " << k + 1
            << " | Yaw: " << x(8)
            << " | Height: " << x(2)
            << " | Yaw rate: " << x(11)
            << "\n";
    }

    // ==========================================
    // 7. Verify time-varying dynamics
    // ==========================================

    double A_difference =
        (A_sequence.front() - A_sequence.back()).norm();

    double B_difference =
        (B_sequence.front() - B_sequence.back()).norm();

    std::cout << "\nA variation: "
              << A_difference << "\n";

    std::cout << "B variation: "
              << B_difference << "\n";

    if (A_difference < 1e-10 ||
        B_difference < 1e-10)
    {
        std::cerr
            << "\nTIME-VARYING DYNAMICS TEST FAILED!\n";

        return 1;
    }

    // ==========================================
    // 8. Construct LTV prediction matrices
    // ==========================================

    PredictionModel prediction(horizon);

    prediction.computeTimeVaryingPredictionMatrices(
        A_sequence,
        B_sequence
    );

    std::cout << "\nP dimensions: "
              << prediction.getP().rows()
              << " x "
              << prediction.getP().cols()
              << "\n";

    std::cout << "H dimensions: "
              << prediction.getH().rows()
              << " x "
              << prediction.getH().cols()
              << "\n";

    std::cout
        << "\nTURNING TRAJECTORY TEST PASSED!\n";

    return 0;
}
