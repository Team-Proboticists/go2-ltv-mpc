#include "propup_ltv_mpc/PredictionModel.hpp"

#include <Eigen/Dense>
#include <vector>
#include <stdexcept>

// ==========================================
// 1. Constructor
// ==========================================

PredictionModel::PredictionModel(int horizon)
    : horizon_(horizon)
{
    if (horizon_ <= 0)
    {
        throw std::invalid_argument(
            "Prediction horizon must be positive"
        );
    }

    P_ = Eigen::MatrixXd::Zero(
        STATE_DIM * horizon_,
        STATE_DIM
    );

    H_ = Eigen::MatrixXd::Zero(
        STATE_DIM * horizon_,
        INPUT_DIM * horizon_
    );
}

// ==========================================
// 2. Time-invariant prediction matrices
// ==========================================

void PredictionModel::computePredictionMatrices(
    const StateMatrix& Ad,
    const InputMatrix& Bd)
{
    const int nx = STATE_DIM;
    const int nu = INPUT_DIM;

    P_.setZero();
    H_.setZero();

    // Initial state transition matrix
    StateMatrix A_power = StateMatrix::Identity();

    for (int k = 0; k < horizon_; ++k)
    {
        // A^(k+1)
        A_power = Ad * A_power;

        // Initial state contribution
        P_.block(
            k * nx,
            0,
            nx,
            nx
        ) = A_power;

        // Control input contributions
        for (int j = 0; j <= k; ++j)
        {
            StateMatrix A_term =
                StateMatrix::Identity();

            // Compute A^(k-j)
            for (int p = 0; p < k - j; ++p)
            {
                A_term = Ad * A_term;
            }

            H_.block(
                k * nx,
                j * nu,
                nx,
                nu
            ) = A_term * Bd;
        }
    }
}

// ==========================================
// 3. Time-varying prediction matrices
// ==========================================

void PredictionModel::computeTimeVaryingPredictionMatrices(
    const std::vector<StateMatrix>& A_sequence,
    const std::vector<InputMatrix>& B_sequence)
{
    // Check sequence lengths
    if (static_cast<int>(A_sequence.size()) != horizon_ ||
        static_cast<int>(B_sequence.size()) != horizon_)
    {
        throw std::invalid_argument(
            "Dynamics sequence length must equal horizon"
        );
    }

    const int nx = STATE_DIM;
    const int nu = INPUT_DIM;

    P_.setZero();
    H_.setZero();

    // Initial state transition matrix
    StateMatrix Phi = StateMatrix::Identity();

    // ==========================================
    // Build time-varying prediction matrices
    // ==========================================

    for (int k = 0; k < horizon_; ++k)
    {
        // --------------------------------------
        // A. State transition contribution
        // --------------------------------------

        Phi = A_sequence[k] * Phi;

        P_.block(
            k * nx,
            0,
            nx,
            nx
        ) = Phi;

        // --------------------------------------
        // B. Previous control contributions
        // --------------------------------------

        for (int j = 0; j < k; ++j)
        {
            H_.block(
                k * nx,
                j * nu,
                nx,
                nu
            ) =
                A_sequence[k] *
                H_.block(
                    (k - 1) * nx,
                    j * nu,
                    nx,
                    nu
                ).eval();
        }

        // --------------------------------------
        // C. Current control contribution
        // --------------------------------------

        H_.block(
            k * nx,
            k * nu,
            nx,
            nu
        ) = B_sequence[k];
    }
}

// ==========================================
// 4. Return prediction matrix P
// ==========================================

const Eigen::MatrixXd&
PredictionModel::getP() const
{
    return P_;
}

// ==========================================
// 5. Return prediction matrix H
// ==========================================

const Eigen::MatrixXd&
PredictionModel::getH() const
{
    return H_;
}
