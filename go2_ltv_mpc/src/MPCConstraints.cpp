#include "propup_ltv_mpc/MPCConstraints.hpp"

#include <limits>
#include <stdexcept>

MPCConstraints::MPCConstraints(
    int horizon,
    double friction_coefficient,
    double max_normal_force
)
    : horizon_(horizon),
      mu_(friction_coefficient),
      fz_max_(max_normal_force)
{
    if (horizon_ <= 0 || mu_ < 0.0 || fz_max_ < 0.0)
    {
        throw std::invalid_argument(
            "Invalid MPC constraint parameters"
        );
    }

    const int rows =
        horizon_ * NUM_LEGS * CONSTRAINTS_PER_LEG;

    const int cols = horizon_ * INPUT_DIM;

    C_ = Eigen::MatrixXd::Zero(rows, cols);

    lower_bound_ = Eigen::VectorXd::Zero(rows);
    upper_bound_ = Eigen::VectorXd::Zero(rows);
}

void MPCConstraints::buildConstraints(
    const std::array<int, 4>& stance_state
)
{
    const double INF = std::numeric_limits<double>::infinity();

    C_.setZero();

    for (int k = 0; k < horizon_; ++k)
    {
        for (int leg = 0; leg < NUM_LEGS; ++leg)
        {
            const int row =
                (k * NUM_LEGS + leg) * CONSTRAINTS_PER_LEG;

            const int col =
                k * INPUT_DIM + leg * FORCE_DIM;

            // Constraint 1: fx - mu*fz <= 0
            C_(row, col) = 1.0;
            C_(row, col + 2) = -mu_;

            // Constraint 2: -fx - mu*fz <= 0
            C_(row + 1, col) = -1.0;
            C_(row + 1, col + 2) = -mu_;

            // Constraint 3: fy - mu*fz <= 0
            C_(row + 2, col + 1) = 1.0;
            C_(row + 2, col + 2) = -mu_;

            // Constraint 4: -fy - mu*fz <= 0
            C_(row + 3, col + 1) = -1.0;
            C_(row + 3, col + 2) = -mu_;

            // Constraint 5: normal force
            C_(row + 4, col + 2) = 1.0;

            for (int j = 0; j < 4; ++j)
            {
                lower_bound_(row + j) = -INF;
                upper_bound_(row + j) = 0.0;
            }

            if (stance_state[leg] == 1)
            {
                // Stance leg:
                // 0 <= fz <= fz_max
                lower_bound_(row + 4) = 0.0;
                upper_bound_(row + 4) = fz_max_;
            }
            else
            {
                // Swing leg:
                // fz = 0, which also forces
                // fx = fy = 0 through friction constraints
                lower_bound_(row + 4) = 0.0;
                upper_bound_(row + 4) = 0.0;
            }
        }
    }
}

const Eigen::MatrixXd&
MPCConstraints::getC() const
{
    return C_;
}

const Eigen::VectorXd&
MPCConstraints::getUpperBound() const
{
    return upper_bound_;
}

const Eigen::VectorXd&
MPCConstraints::getLowerBound() const
{
    return lower_bound_;
}
