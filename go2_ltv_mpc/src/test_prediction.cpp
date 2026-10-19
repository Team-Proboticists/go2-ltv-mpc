#include <iostream>
#include <Eigen/Dense>
#include <cmath>

#include "propup_ltv_mpc/PredictionModel.hpp"

int main()
{
    // ==========================================
    // 1. Define prediction horizon
    // ==========================================

    int N = 10;

    PredictionModel prediction(N);

    // ==========================================
    // 2. Create simple test dynamics
    // ==========================================

    PredictionModel::StateMatrix Ad =
        PredictionModel::StateMatrix::Identity();

    PredictionModel::InputMatrix Bd =
        PredictionModel::InputMatrix::Zero();

    double dt = 0.02;

    // Simple position-velocity dynamics
    // x_next = x + dt * v
    Ad(0, 3) = dt;

    // Control input affects velocity
    Bd(3, 0) = dt;

    // ==========================================
    // 3. Construct prediction matrices
    // ==========================================

    prediction.computePredictionMatrices(Ad, Bd);

    Eigen::MatrixXd P = prediction.getP();
    Eigen::MatrixXd H = prediction.getH();

    // ==========================================
    // 4. Check matrix dimensions
    // ==========================================

    std::cout << "P dimensions: "
              << P.rows() << " x "
              << P.cols() << std::endl;

    std::cout << "H dimensions: "
              << H.rows() << " x "
              << H.cols() << std::endl;

    // ==========================================
    // 5. Define initial robot state
    // ==========================================

    Eigen::VectorXd x0 =
        Eigen::VectorXd::Zero(13);

    x0(0) = 1.0;
    x0(3) = 2.0;

    // ==========================================
    // 6. Define control sequence
    // ==========================================

    Eigen::VectorXd U =
        Eigen::VectorXd::Zero(N * 12);

    // Apply constant test input
    for (int k = 0; k < N; ++k)
    {
        U(k * 12) = 1.0;
    }

    // ==========================================
    // 7. Predict using P and H
    // ==========================================

    Eigen::VectorXd X = P * x0 + H * U;

    // ==========================================
    // 8. Independently simulate dynamics
    // ==========================================

    Eigen::VectorXd x = x0;

    Eigen::VectorXd X_direct =
        Eigen::VectorXd::Zero(N * 13);

    for (int k = 0; k < N; ++k)
    {
        Eigen::VectorXd uk =
            U.segment(k * 12, 12);

        x = Ad * x + Bd * uk;

        X_direct.segment(k * 13, 13) = x;
    }

    // ==========================================
    // 9. Compare predictions
    // ==========================================

    double error = (X - X_direct).norm();

    std::cout << "\nPrediction error: "
              << error << std::endl;

    std::cout << "\nFinal predicted position: "
              << X((N - 1) * 13)
              << std::endl;

    std::cout << "Final predicted velocity: "
              << X((N - 1) * 13 + 3)
              << std::endl;

    if (error < 1e-10)
    {
        std::cout << "\nTEST PASSED!" << std::endl;
    }
    else
    {
        std::cout << "\nTEST FAILED!" << std::endl;
        return 1;
    }

    return 0;
}
