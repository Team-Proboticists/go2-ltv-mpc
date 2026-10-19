#include "propup_ltv_mpc/MPCCost.hpp"

#include <stdexcept>

// Constructor
MPCCost::MPCCost()
{
}

// Set the weighting matrices
void MPCCost::setWeights(
    const Matrix& Q,
    const Matrix& R)
{
    if (Q.rows() != Q.cols() ||
        R.rows() != R.cols())
    {
        throw std::invalid_argument(
            "Q and R must be square matrices");
    }

    Q_ = Q;
    R_ = R;
}

// Compute quadratic programming cost
void MPCCost::computeCost(
    const Matrix& P,
    const Matrix& H,
    const Vector& x0,
    const Vector& X_ref)
{
    // Check dimensions
    if (P.cols() != x0.size() ||
        P.rows() != H.rows() ||
        X_ref.size() != P.rows() ||
        Q_.rows() != P.rows() ||
        R_.rows() != H.cols())
    {
        throw std::invalid_argument(
            "Incompatible MPC matrix dimensions");
    }

    // Predicted state error without control
    Vector error = P * x0 - X_ref;

    // Quadratic programming Hessian
    hessian_ = 2.0 * (
        H.transpose() * Q_ * H + R_
    );

    // Quadratic programming gradient
    gradient_ = 2.0 * H.transpose() * Q_ * error;
}

// Return Hessian
const MPCCost::Matrix& MPCCost::getHessian() const
{
    return hessian_;
}

// Return gradient
const MPCCost::Vector& MPCCost::getGradient() const
{
    return gradient_;
}
// ==========================================
// MPC cost with nominal control reference
// ==========================================

// ==========================================
// MPC cost with nominal control reference
// ==========================================

void MPCCost::computeCost(
    const Matrix& P,
    const Matrix& H,
    const Vector& x0,
    const Vector& X_ref,
    const Vector& U_ref)
{
    // Check matrix dimensions
    if (P.cols() != x0.size() ||
        P.rows() != H.rows() ||
        X_ref.size() != P.rows() ||
        U_ref.size() != H.cols() ||
        Q_.rows() != P.rows() ||
        Q_.cols() != P.rows() ||
        R_.rows() != H.cols() ||
        R_.cols() != H.cols())
    {
        throw std::invalid_argument(
            "Incompatible MPC matrix dimensions");
    }

    // State prediction error without control
    Vector error = P * x0 - X_ref;

    // Hessian
    hessian_ =
        2.0 * (H.transpose() * Q_ * H + R_);

    // Gradient with nominal control reference
    gradient_ =
        2.0 * H.transpose() * Q_ * error
        - 2.0 * R_ * U_ref;
}
