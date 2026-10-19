#include "propup_ltv_mpc/QPSolver.hpp"

#include <Eigen/Dense>
#include <iostream>
#include <cmath>

int main()
{
    // Define a simple quadratic optimization problem
    Eigen::MatrixXd G(2, 2);

    G << 4.0, 1.0,
         1.0, 2.0;

    Eigen::VectorXd g(2);

    g << -1.0, -1.0;

    // Create QP solver
    QPSolver solver;

    // Solve optimization problem
    bool success = solver.solve(G, g);

    if (!success)
    {
        std::cerr << "QP solver failed!" << std::endl;
        return 1;
    }

    // Retrieve optimal solution
    Eigen::VectorXd U = solver.getSolution();

    std::cout << "\nOptimal solution:\n"
              << U << std::endl;

    // Calculate optimality residual
    double residual = (G * U + g).norm();

    std::cout << "\nOptimality residual: "
              << residual << std::endl;

    // Calculate objective value
    double cost =
        0.5 * U.dot(G * U) + g.dot(U);

    std::cout << "Optimal cost: "
              << cost << std::endl;

    // Verify known analytical solution
    Eigen::Vector2d expected;
    expected << 1.0 / 7.0, 3.0 / 7.0;

    double solution_error = (U - expected).norm();

    std::cout << "Solution error: "
              << solution_error << std::endl;

    if (residual < 1e-8 &&
        solution_error < 1e-8)
    {
        std::cout << "\nTEST PASSED!" << std::endl;
        return 0;
    }

    std::cout << "\nTEST FAILED!" << std::endl;
    return 1;
}
