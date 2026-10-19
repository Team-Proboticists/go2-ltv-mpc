#ifndef PREDICTION_MODEL_HPP
#define PREDICTION_MODEL_HPP

#include <Eigen/Dense>
#include <vector>

class PredictionModel
{
public:

    // ==========================================
    // 1. State and input dimensions
    // ==========================================

    static constexpr int STATE_DIM = 13;
    static constexpr int INPUT_DIM = 12;

    // ==========================================
    // 2. Matrix definitions
    // ==========================================

    using StateMatrix =
        Eigen::Matrix<double, STATE_DIM, STATE_DIM>;

    using InputMatrix =
        Eigen::Matrix<double, STATE_DIM, INPUT_DIM>;

    // ==========================================
    // 3. Constructor
    // ==========================================

    explicit PredictionModel(int horizon);

    // ==========================================
    // 4. Time-invariant prediction matrices
    // ==========================================

    void computePredictionMatrices(
        const StateMatrix& Ad,
        const InputMatrix& Bd
    );

    // ==========================================
    // 5. Time-varying prediction matrices
    // ==========================================

    void computeTimeVaryingPredictionMatrices(
        const std::vector<StateMatrix>& A_sequence,
        const std::vector<InputMatrix>& B_sequence
    );

    // ==========================================
    // 6. Access prediction matrices
    // ==========================================

    const Eigen::MatrixXd& getP() const;

    const Eigen::MatrixXd& getH() const;

private:

    // Prediction horizon
    int horizon_;

    // State prediction matrix
    Eigen::MatrixXd P_;

    // Control prediction matrix
    Eigen::MatrixXd H_;
};

#endif
