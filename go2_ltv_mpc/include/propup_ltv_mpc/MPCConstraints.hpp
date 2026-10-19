#ifndef MPC_CONSTRAINTS_HPP
#define MPC_CONSTRAINTS_HPP

#include <Eigen/Dense>
#include <array>

class MPCConstraints
{
public:
    static constexpr int NUM_LEGS = 4;
    static constexpr int FORCE_DIM = 3;
    static constexpr int INPUT_DIM = 12;

    // Five inequalities per foot per prediction step:
    // fx - mu*fz <= 0
    // -fx - mu*fz <= 0
    // fy - mu*fz <= 0
    // -fy - mu*fz <= 0
    // fz <= fz_max

    static constexpr int CONSTRAINTS_PER_LEG = 5;

    MPCConstraints(
        int horizon,
        double friction_coefficient,
        double max_normal_force
    );

    // Construct constraint matrices for the MPC horizon
    void buildConstraints(
        const std::array<int, 4>& stance_state
    );

    // Access constraint matrix
    const Eigen::MatrixXd& getC() const;

    // Access upper bounds
    const Eigen::VectorXd& getUpperBound() const;

    // Access lower bounds
    const Eigen::VectorXd& getLowerBound() const;

private:
    int horizon_;

    double mu_;
    double fz_max_;

    Eigen::MatrixXd C_;

    Eigen::VectorXd lower_bound_;
    Eigen::VectorXd upper_bound_;
};

#endif
