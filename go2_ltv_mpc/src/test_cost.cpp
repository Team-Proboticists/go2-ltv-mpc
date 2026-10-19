#include "propup_ltv_mpc/MPCCost.hpp"

#include <Eigen/Dense>
#include <iostream>
#include <cmath>

int main()
{
    constexpr int nx = 13;
    constexpr int nu = 12;
    constexpr int N = 10;

    // Prediction matrices
    Eigen::MatrixXd P =
        Eigen::MatrixXd::Zero(N * nx, nx);

    Eigen::MatrixXd H =
        Eigen::MatrixXd::Zero(N * nx, N * nu);

    // Simple test system:
    // x(k+1) = x(k) + B*u(k)
    Eigen::MatrixXd B =
        Eigen::MatrixXd::Zero(nx, nu);

    B.block(0, 0, 12, 12).setIdentity();

    // Construct prediction matrices
    for (int i = 0; i < N; ++i)
    {
        P.block(i * nx, 0, nx, nx).setIdentity();

        for (int j = 0; j <= i; ++j)
        {
            H.block(i * nx, j * nu, nx, nu) = B;
        }
    }

    // Initial state
    Eigen::VectorXd x0 =
        Eigen::VectorXd::Zero(nx);

    // Desired states
    Eigen::VectorXd X_ref =
        Eigen::VectorXd::Ones(N * nx);

    // Weighting matrices
    Eigen::MatrixXd Q =
        Eigen::MatrixXd::Identity(N * nx, N * nx);

    Eigen::MatrixXd R =
        0.01 * Eigen::MatrixXd::Identity(N * nu, N * nu);

    // Create MPC cost object
    MPCCost cost;

    cost.setWeights(Q, R);

    cost.computeCost(P, H, x0, X_ref);

    Eigen::MatrixXd G = cost.getHessian();
    Eigen::VectorXd g = cost.getGradient();

    std::cout << "Hessian dimensions: "
              << G.rows() << " x "
              << G.cols() << std::endl;

    std::cout << "Gradient dimensions: "
              << g.rows() << " x "
              << g.cols() << std::endl;

    // Check symmetry
    double symmetry_error =
        (G - G.transpose()).norm();

    std::cout << "Hessian symmetry error: "
              << symmetry_error << std::endl;

    // Check positive definiteness
    Eigen::LLT<Eigen::MatrixXd> llt(G);

    bool positive_definite =
        (llt.info() == Eigen::Success);

    std::cout << "Hessian positive definite: "
              << (positive_definite ? "YES" : "NO")
              << std::endl;

    // Check gradient using finite differences
    Eigen::VectorXd U =
        Eigen::VectorXd::Constant(N * nu, 0.1);

    Eigen::VectorXd analytical_gradient = G * U + g;

    Eigen::VectorXd numerical_gradient(N * nu);

    const double eps = 1e-6;

    auto objective = [&](const Eigen::VectorXd& u)
    {
        Eigen::VectorXd error =
            P * x0 + H * u - X_ref;

        return error.dot(Q * error) + u.dot(R * u);
    };

    for (int i = 0; i < U.size(); ++i)
    {
        Eigen::VectorXd up = U;
        Eigen::VectorXd um = U;

        up(i) += eps;
        um(i) -= eps;

        numerical_gradient(i) =
            (objective(up) - objective(um)) / (2.0 * eps);
    }

    double gradient_error =
        (analytical_gradient - numerical_gradient).norm();

    std::cout << "Gradient error: "
              << gradient_error << std::endl;

    bool passed =
        symmetry_error < 1e-10 &&
        positive_definite &&
        gradient_error < 1e-5;

    if (passed)
    {
        std::cout << "\nTEST PASSED!" << std::endl;
        return 0;
    }

    std::cout << "\nTEST FAILED!" << std::endl;
    return 1;
}
