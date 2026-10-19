#ifndef QP_SOLVER_HPP
#define QP_SOLVER_HPP

#include <Eigen/Dense>

class QPSolver
{
public:
    using Matrix = Eigen::MatrixXd;
    using Vector = Eigen::VectorXd;

    QPSolver();

    // Existing unconstrained solver
    bool solve(
        const Matrix& G,
        const Vector& g
    );

    // New constrained solver
    bool solve(
        const Matrix& G,
        const Vector& g,
        const Matrix& C,
        const Vector& lower_bound,
        const Vector& upper_bound
    );

    const Vector& getSolution() const;

private:
    Vector solution_;
};

#endif
