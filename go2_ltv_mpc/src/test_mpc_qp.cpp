#include "propup_ltv_mpc/QPSolver.hpp"
#include "propup_ltv_mpc/MPCConstraints.hpp"

#include <Eigen/Dense>
#include <iostream>
#include <array>
#include <cmath>

int main()
{
    const int horizon = 10;
    const int num_inputs = 12;
    const int n = horizon * num_inputs;

    const double mu = 0.5;
    const double max_force = 100.0;

    // Legs 0 and 2 are in contact with the ground.
    // Legs 1 and 3 are swinging.
    std::array<int, 4> stance = {1, 0, 1, 0};

    // ------------------------------------------------
    // 1. Build the force constraints
    // ------------------------------------------------

    MPCConstraints constraints(
        horizon,
        mu,
        max_force
    );

    constraints.buildConstraints(stance);

    Eigen::MatrixXd C = constraints.getC();

    Eigen::VectorXd lower = constraints.getLowerBound();
    Eigen::VectorXd upper = constraints.getUpperBound();

    // ------------------------------------------------
    // 2. Define a simple quadratic objective
    // ------------------------------------------------

    Eigen::MatrixXd G =
        Eigen::MatrixXd::Identity(n, n);

    Eigen::VectorXd desired =
        Eigen::VectorXd::Zero(n);

    for (int k = 0; k < horizon; ++k)
    {
        // Desired forces for stance leg 0
        desired(k * 12 + 0) = 40.0;
        desired(k * 12 + 1) = 10.0;
        desired(k * 12 + 2) = 50.0;

        // Deliberately request force on swing leg 1
        desired(k * 12 + 3) = 20.0;
        desired(k * 12 + 5) = 30.0;

        // Desired forces for stance leg 2
        desired(k * 12 + 6) = -40.0;
        desired(k * 12 + 7) = 10.0;
        desired(k * 12 + 8) = 50.0;
    }

    // Minimize 0.5 * ||U - desired||^2
    // Constant terms can be omitted.

    Eigen::VectorXd g = -desired;

    // ------------------------------------------------
    // 3. Solve constrained QP
    // ------------------------------------------------

    QPSolver solver;

    bool success = solver.solve(
        G,
        g,
        C,
        lower,
        upper
    );

    if (!success)
    {
        std::cerr << "QP solver failed!\n";
        return 1;
    }

    Eigen::VectorXd U = solver.getSolution();

    std::cout << "\nFirst timestep optimized forces:\n";

    for (int leg = 0; leg < 4; ++leg)
    {
        int index = leg * 3;

        std::cout
            << "Leg " << leg << ": "
            << U(index) << " "
            << U(index + 1) << " "
            << U(index + 2) << "\n";
    }

    // ------------------------------------------------
    // 4. Verify constraints
    // ------------------------------------------------

    Eigen::VectorXd CU = C * U;

    double tolerance = 1e-3;

    bool valid =
        (CU.array() >= lower.array() - tolerance).all() &&
        (CU.array() <= upper.array() + tolerance).all();

    if (!valid)
    {
        std::cerr << "\nConstraint violation detected!\n";
        return 1;
    }

    // Verify swing-leg forces are zero.
    for (int k = 0; k < horizon; ++k)
    {
        for (int leg : {1, 3})
        {
            int index = k * 12 + leg * 3;

            if (U.segment(index, 3).norm() > tolerance)
            {
                std::cerr
                    << "Nonzero force on swing leg!\n";

                return 1;
            }
        }
    }

    std::cout << "\nAll MPC constraints satisfied!\n";
    std::cout << "TEST PASSED!\n";

    return 0;
}
