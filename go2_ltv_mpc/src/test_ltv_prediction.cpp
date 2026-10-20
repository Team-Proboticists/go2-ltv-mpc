#include "propup_ltv_mpc/PredictionModel.hpp"

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>

int main()
{
    constexpr int nx = PredictionModel::STATE_DIM;
    constexpr int nu = PredictionModel::INPUT_DIM;

    const int horizon = 10;

    using StateMatrix = PredictionModel::StateMatrix;
    using InputMatrix = PredictionModel::InputMatrix;

    // ==========================================
    // TEST 1: Constant dynamics
    // ==========================================

    std::cout << "TEST 1: Constant dynamics\n";

    StateMatrix A = StateMatrix::Identity();
    InputMatrix B = InputMatrix::Zero();

    // Simple position-velocity relationship
    A(0, 3) = 0.02;
    A(1, 4) = 0.02;
    A(2, 5) = 0.02;

    // Simple force-to-velocity relationship
    B.block<3, 3>(3, 0) =
        Eigen::Matrix3d::Identity() * 0.02;

    PredictionModel constant_model(horizon);

    constant_model.computePredictionMatrices(A, B);

    Eigen::MatrixXd P_constant = constant_model.getP();
    Eigen::MatrixXd H_constant = constant_model.getH();

    std::vector<StateMatrix> A_sequence(
        horizon, A
    );

    std::vector<InputMatrix> B_sequence(
        horizon, B
    );

    PredictionModel varying_model(horizon);

    varying_model.computeTimeVaryingPredictionMatrices(
        A_sequence,
        B_sequence
    );

    Eigen::MatrixXd P_varying = varying_model.getP();
    Eigen::MatrixXd H_varying = varying_model.getH();

    double P_error =
        (P_constant - P_varying).norm();

    double H_error =
        (H_constant - H_varying).norm();

    std::cout << "P error: " << P_error << "\n";
    std::cout << "H error: " << H_error << "\n";

    if (P_error > 1e-10 || H_error > 1e-10)
    {
        std::cerr << "TEST 1 FAILED!\n";
        return 1;
    }

    std::cout << "TEST 1 PASSED!\n\n";

    // ==========================================
    // TEST 2: Time-varying dynamics
    // ==========================================

    std::cout << "TEST 2: Time-varying dynamics\n";

    // Make dynamics different at every timestep
    for (int k = 0; k < horizon; ++k)
    {
        A_sequence[k] = A;
        B_sequence[k] = B;

        A_sequence[k](0, 3) =
            0.02 + 0.001 * k;

        B_sequence[k](3, 0) =
            0.02 + 0.002 * k;
    }

    varying_model.computeTimeVaryingPredictionMatrices(
        A_sequence,
        B_sequence
    );

    // Initial state
    Eigen::VectorXd x0 =
        Eigen::VectorXd::Zero(nx);

    x0(0) = 1.0;
    x0(3) = 0.5;

    // Input sequence
    Eigen::VectorXd U =
        Eigen::VectorXd::Zero(nu * horizon);

    for (int k = 0; k < horizon; ++k)
    {
        U(k * nu) = 2.0;
    }

    // Prediction using P and H
    Eigen::VectorXd X_pred =
        varying_model.getP() * x0 +
        varying_model.getH() * U;

    // Direct step-by-step simulation
    Eigen::VectorXd X_direct =
        Eigen::VectorXd::Zero(nx * horizon);

    Eigen::VectorXd x = x0;

    for (int k = 0; k < horizon; ++k)
    {
        Eigen::VectorXd u =
            U.segment(k * nu, nu);

        x = A_sequence[k] * x +
            B_sequence[k] * u;

        X_direct.segment(k * nx, nx) = x;
    }

    double prediction_error =
        (X_pred - X_direct).norm();

    std::cout << "Prediction error: "
              << prediction_error << "\n";

    if (prediction_error > 1e-10)
    {
        std::cerr << "TEST 2 FAILED!\n";
        return 1;
    }

    std::cout << "TEST 2 PASSED!\n\n";

    std::cout << "ALL LTV PREDICTION TESTS PASSED!\n";

    return 0;
}
