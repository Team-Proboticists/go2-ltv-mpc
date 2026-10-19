#ifndef LTV_MPC_HPP
#define LTV_MPC_HPP

#include <Eigen/Dense>
#include <array>

class LTV_MPC
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

    // 13 x 13 state matrix
    using StateMatrix =
        Eigen::Matrix<double, STATE_DIM, STATE_DIM>;

    // 13 x 12 input matrix
    using InputMatrix =
        Eigen::Matrix<double, STATE_DIM, INPUT_DIM>;

    // 13 x 1 state vector
    using StateVector =
        Eigen::Matrix<double, STATE_DIM, 1>;

    // ==========================================
    // 3. Constructor
    // ==========================================

    LTV_MPC();

    // ==========================================
    // 4. Set physical robot parameters
    // ==========================================

    void setRobotParameters(
        double mass,
        const Eigen::Matrix3d& inertia
    );

    // ==========================================
    // 5. Continuous-time dynamics
    // ==========================================

    void computeContinuousDynamics(
        const StateVector& state,
        const std::array<Eigen::Vector3d, 4>& foot_positions
    );

    // Access continuous-time matrices
    const StateMatrix& getA() const;

    const InputMatrix& getB() const;

    // ==========================================
    // 6. Discrete-time dynamics
    // ==========================================

    void discretizeDynamics(double dt);

    // Access discrete-time matrices
    const StateMatrix& getAd() const;

    const InputMatrix& getBd() const;

private:

    // ==========================================
    // 7. Robot physical parameters
    // ==========================================

    double mass_;

    Eigen::Matrix3d inertia_;

    // ==========================================
    // 8. Continuous-time matrices
    // ==========================================

    StateMatrix A_c_;

    InputMatrix B_c_;

    // ==========================================
    // 9. Discrete-time matrices
    // ==========================================

    StateMatrix A_d_;

    InputMatrix B_d_;

    // ==========================================
    // 10. Skew-symmetric matrix function
    // ==========================================

    Eigen::Matrix3d skew(
        const Eigen::Vector3d& vector
    );

};

#endif
