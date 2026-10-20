#include "propup_ltv_mpc/LTV_MPC.hpp"
#include "propup_ltv_mpc/PredictionModel.hpp"
#include "propup_ltv_mpc/MPCCost.hpp"
#include "propup_ltv_mpc/MPCConstraints.hpp"
#include "propup_ltv_mpc/QPSolver.hpp"

#include <Eigen/Dense>
#include <array>
#include <iostream>
#include <cmath>

int main()
{
    const int N = 10;
    const double dt = 0.02;
    const double mass = 12.0;

    // ==========================================
    // 1. Robot dynamics
    // ==========================================

    LTV_MPC dynamics;

    Eigen::Matrix3d inertia = Eigen::Matrix3d::Zero();
    inertia.diagonal() << 0.1, 0.25, 0.3;

    dynamics.setRobotParameters(mass, inertia);

    LTV_MPC::StateVector x0 =
        LTV_MPC::StateVector::Zero();

    x0(2) = 0.30;
    x0(12) = 9.81;

    std::array<Eigen::Vector3d, 4> feet = {
        Eigen::Vector3d( 0.20,  0.10, -0.30),
        Eigen::Vector3d( 0.20, -0.10, -0.30),
        Eigen::Vector3d(-0.20,  0.10, -0.30),
        Eigen::Vector3d(-0.20, -0.10, -0.30)
    };

    dynamics.computeContinuousDynamics(x0, feet);
    dynamics.discretizeDynamics(dt);

    // ==========================================
    // 2. Prediction matrices
    // ==========================================

    PredictionModel prediction(N);

    prediction.computePredictionMatrices(
        dynamics.getAd(),
        dynamics.getBd()
    );

    Eigen::MatrixXd P = prediction.getP();
    Eigen::MatrixXd H = prediction.getH();

    // ==========================================
    // 3. Nominal support forces
    // ==========================================

    Eigen::VectorXd U_ref =
        Eigen::VectorXd::Zero(12 * N);

    const double nominal_fz = mass * 9.81 / 4.0;

    for (int k = 0; k < N; ++k)
    {
        for (int leg = 0; leg < 4; ++leg)
        {
            U_ref(k * 12 + leg * 3 + 2) =
                nominal_fz;
        }
    }

    // Reference trajectory consistent with
    // nominal standing forces
    Eigen::VectorXd X_ref = P * x0 + H * U_ref;

    // ==========================================
    // 4. MPC cost
    // ==========================================

    Eigen::MatrixXd Q =
        Eigen::MatrixXd::Identity(13 * N, 13 * N);

    Eigen::MatrixXd R =
        0.001 *
        Eigen::MatrixXd::Identity(12 * N, 12 * N);

    MPCCost cost;
    cost.setWeights(Q, R);

    cost.computeCost(
        P, H, x0, X_ref, U_ref
    );

    // ==========================================
    // 5. Contact constraints
    // ==========================================

    MPCConstraints constraints(
        N,
        0.5,    // friction coefficient
        100.0   // maximum vertical force
    );

    std::array<int, 4> stance = {1, 1, 1, 1};

    constraints.buildConstraints(stance);

    // ==========================================
    // 6. Solve constrained QP
    // ==========================================

    QPSolver solver;

    bool success = solver.solve(
        cost.getHessian(),
        cost.getGradient(),
        constraints.getC(),
        constraints.getLowerBound(),
        constraints.getUpperBound()
    );

    if (!success)
    {
        std::cerr << "OSQP TEST FAILED!\n";
        return 1;
    }

    Eigen::VectorXd U = solver.getSolution();

    // ==========================================
    // 7. Display optimized forces
    // ==========================================

    std::cout << "\nNominal Fz per foot: "
              << nominal_fz << " N\n\n";

    for (int leg = 0; leg < 4; ++leg)
    {
        int i = leg * 3;

        std::cout
            << "Leg " << leg + 1
            << " | Fx: " << U(i)
            << " | Fy: " << U(i + 1)
            << " | Fz: " << U(i + 2)
            << " N\n";
    }

    // ==========================================
    // 8. Check solution
    // ==========================================

    double max_force_error = 0.0;

    for (int leg = 0; leg < 4; ++leg)
    {
        int i = leg * 3;

        max_force_error = std::max(
            max_force_error,
            std::abs(U(i + 2) - nominal_fz)
        );
    }

    double constraint_violation = 0.0;

    Eigen::VectorXd CU = constraints.getC() * U;

    for (int i = 0; i < CU.size(); ++i)
    {
        constraint_violation = std::max(
            constraint_violation,
            std::max(
                0.0,
                CU(i) - constraints.getUpperBound()(i)
            )
        );

        constraint_violation = std::max(
            constraint_violation,
            std::max(
                0.0,
                constraints.getLowerBound()(i) - CU(i)
            )
        );
    }

    std::cout << "\nMaximum vertical force error: "
              << max_force_error << " N\n";

    std::cout << "Maximum constraint violation: "
              << constraint_violation << "\n";

    if (max_force_error < 0.1 &&
        constraint_violation < 1e-4)
    {
        std::cout
            << "\nNOMINAL OSQP TEST PASSED!\n";

        return 0;
    }

    std::cerr
        << "\nNOMINAL OSQP TEST FAILED!\n";

    return 1;
}
