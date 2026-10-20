#include "propup_ltv_mpc/LTV_MPC.hpp"
#include "propup_ltv_mpc/PredictionModel.hpp"

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <array>
#include <cmath>

int main()
{
    const int horizon = 10;
    const double dt = 0.02;

    using StateMatrix = LTV_MPC::StateMatrix;
    using InputMatrix = LTV_MPC::InputMatrix;
    using StateVector = LTV_MPC::StateVector;

    // ==========================================
    // 1. Robot parameters
    // ==========================================

    const double mass = 12.0;

    Eigen::Matrix3d inertia =
        Eigen::Matrix3d::Zero();

    inertia.diagonal() << 0.1, 0.25, 0.3;

    LTV_MPC dynamics;

    dynamics.setRobotParameters(mass, inertia);

    // ==========================================
    // 2. Initial robot state
    // ==========================================

    StateVector x0 = StateVector::Zero();

    x0(2) = 0.30;
    x0(12) = 9.81;

    // ==========================================
    // 3. Foot positions relative to COM
    // ==========================================

    std::array<Eigen::Vector3d, 4> feet;

    feet[0] = Eigen::Vector3d( 0.20,  0.10, -0.30);
    feet[1] = Eigen::Vector3d( 0.20, -0.10, -0.30);
    feet[2] = Eigen::Vector3d(-0.20,  0.10, -0.30);
    feet[3] = Eigen::Vector3d(-0.20, -0.10, -0.30);

    // ==========================================
    // 4. Allocate time-varying matrices
    // ==========================================

    std::vector<StateMatrix> A_sequence;
    std::vector<InputMatrix> B_sequence;

    A_sequence.reserve(horizon);
    B_sequence.reserve(horizon);

    // ==========================================
    // 5. Generate time-varying dynamics
    // ==========================================

    for (int k = 0; k < horizon; ++k)
    {
        StateVector predicted_state = x0;

        // Illustrative predicted yaw trajectory
        predicted_state(8) = 0.05 * k;

        // Illustrative foot geometry variation
        auto predicted_feet = feet;

        for (int leg = 0; leg < 4; ++leg)
        {
            predicted_feet[leg](0) +=
                0.002 * k;
        }

        dynamics.computeContinuousDynamics(
            predicted_state,
            predicted_feet
        );

        dynamics.discretizeDynamics(dt);

        A_sequence.push_back(dynamics.getAd());
        B_sequence.push_back(dynamics.getBd());
    }

    // ==========================================
    // 6. Construct LTV prediction matrices
    // ==========================================

    PredictionModel prediction(horizon);

    prediction.computeTimeVaryingPredictionMatrices(
        A_sequence,
        B_sequence
    );

    const Eigen::MatrixXd& P = prediction.getP();
    const Eigen::MatrixXd& H = prediction.getH();

    std::cout << "P dimensions: "
              << P.rows() << " x "
              << P.cols() << "\n";

    std::cout << "H dimensions: "
              << H.rows() << " x "
              << H.cols() << "\n";

    // ==========================================
    // 7. Check that dynamics actually vary
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
        std::cerr << "Dynamics did not vary as expected!\n";
        return 1;
    }

    // ==========================================
    // 8. Verify predictions
    // ==========================================

    Eigen::VectorXd U =
        Eigen::VectorXd::Zero(12 * horizon);

    // Apply approximately weight-supporting forces
    for (int k = 0; k < horizon; ++k)
    {
        for (int leg = 0; leg < 4; ++leg)
        {
            U(k * 12 + leg * 3 + 2) =
                mass * 9.81 / 4.0;
        }
    }

    Eigen::VectorXd X_pred =
        P * x0 + H * U;

    StateVector x = x0;

    Eigen::VectorXd X_direct =
        Eigen::VectorXd::Zero(13 * horizon);

    for (int k = 0; k < horizon; ++k)
    {
        Eigen::Matrix<double, 12, 1> u =
            U.segment<12>(k * 12);

        x = A_sequence[k] * x +
            B_sequence[k] * u;

        X_direct.segment<13>(k * 13) = x;
    }

    double error =
        (X_pred - X_direct).norm();

    std::cout << "\nPrediction error: "
              << error << "\n";

    if (error > 1e-10)
    {
        std::cerr << "TEST FAILED!\n";
        return 1;
    }

    std::cout << "\nLTV DYNAMICS TEST PASSED!\n";

    return 0;
}
