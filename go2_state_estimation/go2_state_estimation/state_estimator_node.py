import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray

class StateEstimatorNode(Node):
    def __init__(self):
        super().__init__('state_estimator')
        
        # PART 1: Subscribers
        self.imu_subscriber = self.create_subscription( 
            Imu, 
            '/trunk_imu', 
            self.imu_callback, 
            10 
        )

        self.joint_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )

        # PART 2: Publisher
        self.state_publisher = self.create_publisher(
            Float64MultiArray, 
            '/robot_state_estimated', 
            10
        )

        # PART 3: Timer
        self.timer = self.create_timer(0.02, self.timer_callback)
        self.get_logger().info('State estimator booted (Paper Order: [p, v, Θ, ω, g])')

        # PART 4: Matrices & Memory
        # Order: [px, py, pz, vx, vy, vz, roll, pitch, yaw, wx, wy, wz, g]
        self.x_ = np.zeros((13, 1))
        self.x_[2, 0] = 0.2    # Initial nominal height (z0 = 0.2m)
        self.x_[12, 0] = 9.81  # Gravity

        self.P_ = np.eye(13) * 0.01  # Covariance Matrix
        self.Q_ = np.eye(13) * 0.001 # Process Noise
        self.R_ = np.eye(6) * 0.05   # Measurement Noise
        
        self.F_ = np.eye(13) 
        self.H_ = np.zeros((6, 13))
        self.z_ = np.zeros((6, 1))

        # Safety flags and storage
        self.imu_received = False
        self.joint_positions_ = None
        self.joint_velocities_ = None

    # PART 5: EKF Math
    def timer_callback(self):
        # Prevent EKF from running and crashing before first sensor data arrives
        if not self.imu_received:
            return

        # 1. Compute Continuous A_c Matrix (Dynamic Jacobians)
        psi = self.x_[8, 0] # Yaw angle
        
        A_c = np.zeros((13, 13))
        A_c[0:3, 3:6] = np.eye(3)             # p_dot = v
        A_c[3:6, 12] = np.array([[0], [0], [-1]]) # v_dot gravity component
        
        # Theta_dot = Rz * w
        A_c[6:9, 9:12] = np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi),  np.cos(psi), 0],
            [0,            0,           1]
        ])

        # 2. Discretize into F_ Matrix (Forward Euler)
        dt = 0.02
        self.F_ = np.eye(13) + (A_c * dt) 

        # 3. Compute H_ Matrix (Observation Model)
        self.H_ = np.zeros((6, 13))
        self.H_[3:6, 9:12] = np.eye(3) # Gyro measures angular velocity directly
        
        phi = self.x_[6, 0]   # Roll
        theta = self.x_[7, 0] # Pitch
        
        # Accelerometer measures gravity rotated into the body frame
        self.H_[0, 12] = -np.sin(theta)
        self.H_[1, 12] = np.sin(phi) * np.cos(theta)
        self.H_[2, 12] = np.cos(phi) * np.cos(theta)

        # --- PREDICT STEP ---
        self.x_ = self.F_ @ self.x_  # system model calculates state
        self.P_ = self.F_ @ self.P_ @ self.F_.T + self.Q_  # measure of uncertainty
        
        # --- UPDATE STEP ---
        y = self.z_ - self.H_ @ self.x_  # residual
        S = self.H_ @ self.P_ @ self.H_.T + self.R_  # innovation covariance
        
        K = self.P_ @ self.H_.T @ np.linalg.inv(S)  # kalman gain eq
        self.x_ = self.x_ + K @ y  # update state
        self.P_ = (np.eye(13) - K @ self.H_) @ self.P_  # decrease variance

        # KINEMATIC GROUND-CLAMPING CONSTRAINT
        if self.joint_positions_ is not None:
            self.x_[2, 0] = 0.2  

        # Publish Estimated State
        msg = Float64MultiArray()
        msg.data = self.x_.flatten().tolist()
        self.state_publisher.publish(msg)

    # PART 6: Callbacks
    def imu_callback(self, msg):
        # Extract exactly 3 numbers for linear acceleration
        self.z_[0:3, 0] = [msg.linear_acceleration.x, 
                           msg.linear_acceleration.y, 
                           msg.linear_acceleration.z]
        # Extract exactly 3 numbers for angular velocity
        self.z_[3:6, 0] = [msg.angular_velocity.x, 
                           msg.angular_velocity.y, 
                           msg.angular_velocity.z]
        self.imu_received = True

    def joint_callback(self, msg):
        self.joint_positions_ = np.array(msg.position)
        self.joint_velocities_ = np.array(msg.velocity)

# PART 7: Main
def main(args=None):
    rclpy.init(args=args)
    node = StateEstimatorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass 
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
