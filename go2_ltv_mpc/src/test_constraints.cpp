#include "propup_ltv_mpc/MPCConstraints.hpp"

#include <Eigen/Dense>
#include <iostream>
#include <array>
#include <cmath>

int main()
{
    // MPC parameters
    int horizon = 10;
    double mu = 0.5;
    double fz_max = 100.0;

    MPCConstraints constraints(horizon, mu, fz_max);

    // Leg states:
    // 1 = stance (on ground)
    // 0 = swing (in air)
    std::array<int, 4> stance = {1, 0, 1, 0};

    constraints.buildConstraints(stance);

    const auto& C = constraints.getC();
    const auto& lower = constraints.getLowerBound();
    const auto& upper = constraints.getUpperBound();

    std::cout << "C dimensions: "
              << C.rows() << " x " << C.cols() << "\n";

    std::cout << "Lower bound dimensions: "
              << lower.size() << "\n";

    std::cout << "Upper bound dimensions: "
              << upper.size() << "\n";

    // Test forces for one time step:
    // Leg 0: stance, valid forces
    // Leg 1: swing, zero forces
    // Leg 2: stance, valid forces
    // Leg 3: swing, zero forces

    Eigen::VectorXd U = Eigen::VectorXd::Zero(120);

    for (int k = 0; k < horizon; ++k)
    {
        // Leg 0
        U(k * 12 + 0) = 10.0;
        U(k * 12 + 1) = 5.0;
        U(k * 12 + 2) = 50.0;

        // Leg 2
        U(k * 12 + 6) = -10.0;
        U(k * 12 + 7) = 5.0;
        U(k * 12 + 8) = 50.0;
    }

    Eigen::VectorXd result = C * U;

    bool passed = true;

    for (int i = 0; i < result.size(); ++i)
    {
        if (result(i) < lower(i) - 1e-8 ||
            result(i) > upper(i) + 1e-8)
        {
            passed = false;

            std::cout << "Constraint violated at row "
                      << i << "\n";
        }
    }

    std::cout << "\n";

    if (passed)
    {
        std::cout << "All constraints satisfied!\n";
        std::cout << "TEST PASSED!\n";
    }
    else
    {
        std::cout << "TEST FAILED!\n";
    }

    return passed ? 0 : 1;
}
