#include "propup_ltv_mpc/QPSolver.hpp"

#include <Eigen/Dense>
#include <iostream>
#include <cmath>

QPSolver::QPSolver()
{
}

// Solve unconstrained quadratic programming problem
bool QPSolver::solve(
    const Matrix& G,
    const Vector& g)
{
    // Check dimensions
    if (G.rows() != G.cols() ||
        G.rows() != g.size() ||
        G.rows() == 0)
    {
        std::cerr << "Invalid QP dimensions" << std::endl;
        return false;
    }

    // Check numerical values
    if (!G.allFinite() || !g.allFinite())
    {
        std::cerr << "QP contains invalid values" << std::endl;
        return false;
    }

    // Check Hessian symmetry
    if (!G.isApprox(G.transpose(), 1e-10))
    {
        std::cerr << "Hessian is not symmetric" << std::endl;
        return false;
    }

    // Cholesky decomposition
    Eigen::LLT<Matrix> llt(G);

    if (llt.info() != Eigen::Success)
    {
        std::cerr << "Hessian is not positive definite"
                  << std::endl;
        return false;
    }

    // Solve G * U = -g
    solution_ = llt.solve(-g);

    // Verify solution
    if (llt.info() != Eigen::Success ||
        !solution_.allFinite())
    {
        std::cerr << "QP solution failed" << std::endl;
        return false;
    }

    // Check optimality condition
    double residual = (G * solution_ + g).norm();

    double tolerance = 1e-8 * (1.0 + g.norm());

    if (residual > tolerance)
    {
        std::cerr << "QP residual too large: "
                  << residual << std::endl;
        return false;
    }

    std::cout << "QP solved successfully!" << std::endl;

    std::cout << "Optimality residual: "
              << residual << std::endl;

    return true;
}

// Return optimized control inputs
const QPSolver::Vector& QPSolver::getSolution() const
{
    return solution_;
}
// ============================================================
// Constrained QP solver using OSQP 0.6.3
//
// minimize    0.5 * U.transpose() * G * U + g.transpose() * U
// subject to  lower_bound <= C * U <= upper_bound
// ============================================================

#include <osqp/osqp.h>
#include <vector>
#include <algorithm>
#include <memory>

bool QPSolver::solve(
    const Matrix& G,
    const Vector& g,
    const Matrix& C,
    const Vector& lower_bound,
    const Vector& upper_bound)
{
    const int n = static_cast<int>(G.rows());
    const int m = static_cast<int>(C.rows());

    // 1. Validate dimensions and numerical values
    if (n == 0 ||
        G.cols() != n ||
        g.size() != n ||
        C.cols() != n ||
        lower_bound.size() != m ||
        upper_bound.size() != m)
    {
        std::cerr << "Invalid constrained QP dimensions\n";
        return false;
    }

    // Hessian, gradient and constraint matrix must be finite.
// Infinite constraint bounds are permitted.

if (!G.allFinite() ||
    !g.allFinite() ||
    !C.allFinite())
{
    std::cerr << "QP matrices contain invalid values\n";
    return false;
}

// Reject NaN bounds, but permit infinity.
if (lower_bound.array().isNaN().any() ||
    upper_bound.array().isNaN().any())
{
    std::cerr << "QP bounds contain NaN values\n";
    return false;
}

    if ((lower_bound.array() > upper_bound.array()).any())
    {
        std::cerr << "Invalid constraint bounds\n";
        return false;
    }

    if (!G.isApprox(G.transpose(), 1e-10))
    {
        std::cerr << "Hessian is not symmetric\n";
        return false;
    }

    // 2. Convert Eigen matrices to CSC sparse format
    // OSQP stores the Hessian's upper triangle only.

    std::vector<c_int> P_col(n + 1, 0);
    std::vector<c_int> P_row;
    std::vector<c_float> P_val;

    for (int j = 0; j < n; ++j)
    {
        for (int i = 0; i <= j; ++i)
        {
            if (std::abs(G(i, j)) > 1e-12)
            {
                P_row.push_back(i);
                P_val.push_back(G(i, j));
            }
        }

        P_col[j + 1] = static_cast<c_int>(P_val.size());
    }

    std::vector<c_int> A_col(n + 1, 0);
    std::vector<c_int> A_row;
    std::vector<c_float> A_val;

    for (int j = 0; j < n; ++j)
    {
        for (int i = 0; i < m; ++i)
        {
            if (std::abs(C(i, j)) > 1e-12)
            {
                A_row.push_back(i);
                A_val.push_back(C(i, j));
            }
        }

        A_col[j + 1] = static_cast<c_int>(A_val.size());
    }

    // 3. Copy gradient and bounds into OSQP arrays
    std::vector<c_float> q(n);
    std::vector<c_float> l(m);
    std::vector<c_float> u(m);

    for (int i = 0; i < n; ++i)
        q[i] = g(i);

    for (int i = 0; i < m; ++i)
{
    l[i] = std::isinf(lower_bound(i))
        ? (lower_bound(i) < 0 ? -OSQP_INFTY : OSQP_INFTY)
        : lower_bound(i);

    u[i] = std::isinf(upper_bound(i))
        ? (upper_bound(i) < 0 ? -OSQP_INFTY : OSQP_INFTY)
        : upper_bound(i);
}

    // 4. Create sparse matrix descriptors
    // These descriptors reference the vectors above.

    csc* P = csc_matrix(
        n, n,
        static_cast<c_int>(P_val.size()),
        P_val.data(),
        P_row.data(),
        P_col.data()
    );

    csc* A = csc_matrix(
        m, n,
        static_cast<c_int>(A_val.size()),
        A_val.data(),
        A_row.data(),
        A_col.data()
    );

    if (!P || !A)
    {
        std::cerr << "Failed to create OSQP matrices\n";
        if (P) c_free(P);
        if (A) c_free(A);
        return false;
    }

    // 5. Configure the optimization problem
    OSQPData data{};
    data.n = n;
    data.m = m;
    data.P = P;
    data.A = A;
    data.q = q.data();
    data.l = l.data();
    data.u = u.data();

    OSQPSettings settings;
    osqp_set_default_settings(&settings);

    settings.verbose = false;
    settings.warm_start = true;
    settings.max_iter = 10000;
    settings.eps_abs = 1e-5;
    settings.eps_rel = 1e-5;
    settings.polish = true;

    // 6. Initialize OSQP
    OSQPWorkspace* workspace = nullptr;

    c_int exitflag = osqp_setup(
        &workspace,
        &data,
        &settings
    );

    if (exitflag != 0 || workspace == nullptr)
    {
        std::cerr << "OSQP setup failed: "
                  << exitflag << "\n";

        if (workspace)
            osqp_cleanup(workspace);

        c_free(P);
        c_free(A);
        return false;
    }

    // 7. Solve
    exitflag = osqp_solve(workspace);

    bool success =
        exitflag == 0 &&
        workspace->info != nullptr &&
        workspace->info->status_val == OSQP_SOLVED &&
        workspace->solution != nullptr &&
        workspace->solution->x != nullptr;

    if (success)
    {
        Vector candidate(n);

        for (int i = 0; i < n; ++i)
            candidate(i) = workspace->solution->x[i];

        // 8. Verify the numerical solution
        Vector constraint_values = C * candidate;

        double max_violation = 0.0;

for (int i = 0; i < m; ++i)
{
    if (std::isfinite(lower_bound(i)))
    {
        max_violation = std::max(
            max_violation,
            lower_bound(i) - constraint_values(i)
        );
    }

    if (std::isfinite(upper_bound(i)))
    {
        max_violation = std::max(
            max_violation,
            constraint_values(i) - upper_bound(i)
        );
    }
}

max_violation = std::max(0.0, max_violation);


        if (candidate.allFinite() &&
            max_violation <= 1e-3)
        {
            solution_ = candidate;

            std::cout << "Constrained QP solved successfully!\n";
            std::cout << "Maximum constraint violation: "
                      << max_violation << "\n";
        }
        else
        {
            std::cerr << "OSQP solution failed verification\n";
            success = false;
        }
    }
    else
    {
        std::cerr << "OSQP failed to solve the QP\n";

        if (workspace->info)
        {
            std::cerr << "Solver status: "
                      << workspace->info->status << "\n";
        }
    }

    // 9. Release resources
    osqp_cleanup(workspace);
    c_free(P);
    c_free(A);

    return success;
}
