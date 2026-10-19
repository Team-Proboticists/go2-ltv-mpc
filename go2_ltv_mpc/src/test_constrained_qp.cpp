#include "propup_ltv_mpc/QPSolver.hpp"

#include <Eigen/Dense>
#include <iostream>
#include <cmath>

int main()
{
    // Number of optimization variables
    const int n = 2;

    // Number of constraints
    const int m = 2;

    // Hessian matrix
    Eigen::MatrixXd G(n, n);

    G << 1.0, 0.0,
         0.0, 1.0;

    // Gradient vector
    Eigen::VectorXd g(n);

    g << -3.0, -4.0;

    // Constraint matrix
    Eigen::MatrixXd C(m, n);

    C << 1.0, 0.0,
         0.0, 1.0;

    // Lower bounds
    Eigen::VectorXd lower(m);

    lower << 0.0, 0.0;

    // Upper bounds
    Eigen::VectorXd upper(m);

    upper << 2.0, 3.0;

    // Create QP solver
    QPSolver solver;

    // Solve constrained optimization
    bool success = solver.solve(
        G,
        g,
        C,
        lower,
        upper
    );

    if (!success)
    {
        std::cerr << "TEST FAILED: Solver error\n";
        return 1;
    }

    // Extract solution
    Eigen::VectorXd U = solver.getSolution();

    std::cout << "\nOptimal solution:\n";
    std::cout << U << "\n";

    // Expected solution
    Eigen::VectorXd expected(n);

    expected << 2.0, 3.0;

    // Calculate error
    double error = (U - expected).norm();

    std::cout << "\nSolution error: "
              << error << "\n";

    // Verify constraints
    Eigen::VectorXd CU = C * U;

    bool constraints_satisfied =
        (CU.array() >= (lower.array() - 1e-3)).all() &&
        (CU.array() <= (upper.array() + 1e-3)).all();

    if (error < 1e-3 && constraints_satisfied)
    {
        std::cout << "\nTEST PASSED!\n";
        return 0;
    }

    std::cout << "\nTEST FAILED!\n";
    return 1;
}
