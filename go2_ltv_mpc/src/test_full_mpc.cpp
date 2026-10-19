#include "propup_ltv_mpc/LTV_MPC.hpp"
#include "propup_ltv_mpc/PredictionModel.hpp"
#include "propup_ltv_mpc/MPCCost.hpp"
#include "propup_ltv_mpc/MPCConstraints.hpp"
#include "propup_ltv_mpc/QPSolver.hpp"

#include <Eigen/Dense>
#include <iostream>
#include <array>
#include <cmath>

int main()
{
    // ==========================================
    // 1. MPC configuration
    // ==========================================

    const int horizon = 10;
    const double dt = 0.02;

    const int nx = 13;
    const int nu = 12;

    // ==========================================
    // 2. Robot physical parameters
    // ==========================================

    const double mass = 12.0;

    Eigen::Matrix3d inertia =
        Eigen::Matrix3d::Identity();

    inertia(0, 0) = 0.1;
    inertia(1, 1) = 0.25;
    inertia(2, 2) = 0.3;

    // Illustrative parameters only.
    // Replace with verified Go2 parameters later.

    // ==========================================
    // 3. Initial robot state
    // ==========================================

    LTV_MPC::StateVector x0 =
        LTV_MPC::StateVector::Zero();

    // Assumed state ordering:
    // [roll, pitch, yaw,
    //  x, y, z,
    //  wx, wy, wz,
    //  vx, vy, vz,
    //  gravity]

    x0(5) = 0.30;
    x0(12) = -9.81;

    // ==========================================
    // 4. Foot positions relative to COM
    // ==========================================

    std::array<Eigen::Vector3d, 4> feet;

    feet[0] = Eigen::Vector3d( 0.20,  0.10, -0.30);
    feet[1] = Eigen::Vector3d( 0.20, -0.10, -0.30);
    feet[2] = Eigen::Vector3d(-0.20,  0.10, -0.30);
    feet[3] = Eigen::Vector3d(-0.20, -0.10, -0.30);

    // ==========================================
    // 5. Compute robot dynamics
    // ==========================================

    LTV_MPC dynamics;

    dynamics.setRobotParameters(mass, inertia);

    dynamics.computeContinuousDynamics(x0, feet);

    dynamics.discretizeDynamics(dt);

    Eigen::MatrixXd Ad = dynamics.getAd();
    Eigen::MatrixXd Bd = dynamics.getBd();

    std::cout << "Ad: "
              << Ad.rows() << " x "
              << Ad.cols() << "\n";

    std::cout << "Bd: "
              << Bd.rows() << " x "
              << Bd.cols() << "\n";

    // ==========================================
    // 6. Build prediction matrices
    // ==========================================

    PredictionModel prediction(horizon);

    prediction.computePredictionMatrices(
        dynamics.getAd(),
        dynamics.getBd()
    );

    Eigen::MatrixXd P = prediction.getP();
    Eigen::MatrixXd H = prediction.getH();

    std::cout << "P: "
              << P.rows() << " x "
              << P.cols() << "\n";

    std::cout << "H: "
              << H.rows() << " x "
              << H.cols() << "\n";

    // ==========================================
    // 7. Define reference trajectory
    // ==========================================

    Eigen::VectorXd X_ref =
        Eigen::VectorXd::Zero(nx * horizon);

    for (int k = 0; k < horizon; ++k)
    {
        X_ref.segment(k * nx, nx) = x0;
    }

    // ==========================================
    // 8. Define MPC weights
    // ==========================================

    Eigen::MatrixXd Q =
        Eigen::MatrixXd::Identity(nx, nx);

    Eigen::MatrixXd R =
        Eigen::MatrixXd::Identity(nu, nu);

    Q(0, 0) = 50.0;
    Q(1, 1) = 50.0;
    Q(2, 2) = 10.0;

    Q(3, 3) = 20.0;
    Q(4, 4) = 20.0;
    Q(5, 5) = 100.0;

    Q(6, 6) = 5.0;
    Q(7, 7) = 5.0;
    Q(8, 8) = 5.0;

    Q(9, 9) = 10.0;
    Q(10, 10) = 10.0;
    Q(11, 11) = 10.0;

    Q(12, 12) = 1.0;

    R *= 0.001;
    // ==========================================
// Expand Q and R over the prediction horizon
// ==========================================

// Qbar: 130 x 130
Eigen::MatrixXd Qbar =
    Eigen::MatrixXd::Zero(nx * horizon, nx * horizon);

// Rbar: 120 x 120
Eigen::MatrixXd Rbar =
    Eigen::MatrixXd::Zero(nu * horizon, nu * horizon);

// Repeat Q and R along the block diagonals
for (int k = 0; k < horizon; ++k)
{
    Qbar.block(k * nx, k * nx, nx, nx) = Q;

    Rbar.block(k * nu, k * nu, nu, nu) = R;
}

std::cout << "Qbar: "
          << Qbar.rows() << " x "
          << Qbar.cols() << "\n";

std::cout << "Rbar: "
          << Rbar.rows() << " x "
          << Rbar.cols() << "\n";

    // ==========================================
    // 9. Construct MPC cost
    // ==========================================

    MPCCost cost;

    cost.setWeights(Qbar, Rbar);

    cost.computeCost(
        P,
        H,
        x0,
        X_ref
    );

    Eigen::MatrixXd G = cost.getHessian();
    Eigen::VectorXd g = cost.getGradient();

    std::cout << "G: "
              << G.rows() << " x "
              << G.cols() << "\n";

    std::cout << "g: "
              << g.size() << "\n";

    // ==========================================
    // 10. Construct force constraints
    // ==========================================

    const double mu = 0.5;
    const double max_force = 100.0;

    // Use all four legs in stance for the
    // initial standing-controller test.
    std::array<int, 4> stance = {1, 1, 1, 1};

    MPCConstraints constraints(
        horizon,
        mu,
        max_force
    );

    constraints.buildConstraints(stance);

    // ==========================================
    // 11. Solve full MPC optimization
    // ==========================================

    QPSolver solver;

    bool success = solver.solve(
        G,
        g,
        constraints.getC(),
        constraints.getLowerBound(),
        constraints.getUpperBound()
    );

    if (!success)
    {
        std::cerr << "Full MPC optimization failed!\n";
        return 1;
    }

    Eigen::VectorXd U = solver.getSolution();

    // ==========================================
    // 12. Print first timestep forces
    // ==========================================

    std::cout << "\nOptimized ground reaction forces:\n";

    for (int leg = 0; leg < 4; ++leg)
    {
        int index = leg * 3;

        std::cout
            << "Leg " << leg << ": "
            << "Fx = " << U(index)
            << ", Fy = " << U(index + 1)
            << ", Fz = " << U(index + 2)
            << "\n";
    }

    // ==========================================
    // 13. Verify optimized trajectory
    // ==========================================

    Eigen::VectorXd X_pred = P * x0 + H * U;

    std::cout << "\nInitial height: "
              << x0(5) << "\n";

    std::cout << "Predicted height after horizon: "
              << X_pred((horizon - 1) * nx + 5)
              << "\n";

    std::cout << "\nFULL MPC TEST PASSED!\n";

    return 0;
}
