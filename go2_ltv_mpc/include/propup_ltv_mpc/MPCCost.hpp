#ifndef MPC_COST_HPP
#define MPC_COST_HPP

#include <Eigen/Dense>

class MPCCost
{
public:
    using Matrix = Eigen::MatrixXd;
    using Vector = Eigen::VectorXd;

    MPCCost();

    // Set state and control weighting matrices
    void setWeights(
        const Matrix& Q,
        const Matrix& R
    );

    // Construct the quadratic programming cost
    void computeCost(
        const Matrix& P,
        const Matrix& H,
        const Vector& x0,
        const Vector& X_ref
    );
    // Compute MPC cost with nominal control reference
void computeCost(
    const Matrix& P,
    const Matrix& H,
    const Vector& x0,
    const Vector& X_ref,
    const Vector& U_ref
);

    // Access QP matrices
    const Matrix& getHessian() const;
    const Vector& getGradient() const;

private:
    Matrix Q_;
    Matrix R_;

    Matrix hessian_;
    Vector gradient_;
};

#endif
