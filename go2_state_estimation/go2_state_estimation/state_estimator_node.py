import math
import time
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray
from tf2_msgs.msg import TFMessage


def quaternion_to_euler(x, y, z, w):

    norm = math.sqrt(x*x + y*y + z*z + w*w)

    if norm < 1e-8:
        return 0.0, 0.0, 0.0

    x /= norm
    y /= norm
    z /= norm
    w /= norm

    roll = math.atan2(
        2.0 * (w*x + y*z),
        1.0 - 2.0 * (x*x + y*y)
    )

    pitch = math.asin(
        max(-1.0, min(1.0, 2.0 * (w*y - z*x)))
    )

    yaw = math.atan2(
        2.0 * (w*z + x*y),
        1.0 - 2.0 * (y*y + z*z)
    )

    return roll, pitch, yaw


class StateEstimatorNode(Node):

    def __init__(self):

        super().__init__('state_estimator')

        # MPC state vector:
        # [roll, pitch, yaw,
        #  x, y, z,
        #  wx, wy, wz,
        #  vx, vy, vz,
        #  gravity]

        self.x_ = np.zeros(13, dtype=float)
        self.x_[12] = -9.81

        self.imu_received = False
        self.pose_received = False

        # Velocity estimation
        self.previous_position = None
        self.previous_time = None

        # Low-pass filter coefficient
        self.velocity_alpha = 0.2

        # Joint states
        self.joint_positions_ = {}
        self.joint_velocities_ = {}

        # Subscribers
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

        self.pose_subscriber = self.create_subscription(
            TFMessage,
            '/world/empty/dynamic_pose/info',
            self.pose_callback,
            10
        )

        # Publisher
        self.state_publisher = self.create_publisher(
            Float64MultiArray,
            '/robot_state_estimated',
            10
        )

        # Publish at 50 Hz
        self.timer = self.create_timer(
            0.02,
            self.timer_callback
        )

        self.get_logger().info(
            'State estimator started: IMU + Gazebo pose + filtered velocity'
        )

    def imu_callback(self, msg):

        q = msg.orientation

        roll, pitch, yaw = quaternion_to_euler(
            q.x, q.y, q.z, q.w
        )

        # Orientation
        self.x_[0:3] = [roll, pitch, yaw]

        # Angular velocity
        self.x_[6:9] = [
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ]

        self.imu_received = True

    def joint_callback(self, msg):

        self.joint_positions_ = dict(
            zip(msg.name, msg.position)
        )

        self.joint_velocities_ = dict(
            zip(msg.name, msg.velocity)
        )

    def pose_callback(self, msg):

        for transform in msg.transforms:

            if transform.child_frame_id != 'go2_robot':
                continue

            p = transform.transform.translation

            current_position = np.array(
                [p.x, p.y, p.z],
                dtype=float
            )

            # Monotonic wall-clock time
            current_time = time.monotonic()

            if self.previous_position is not None:

                dt = current_time - self.previous_time

                if 0.001 < dt < 0.2:

                    # Finite-difference velocity
                    raw_velocity = (
                        current_position - self.previous_position
                    ) / dt

                    # Previous filtered velocity
                    previous_velocity = self.x_[9:12].copy()

                    # Low-pass filtering
                    filtered_velocity = (
                        self.velocity_alpha * raw_velocity
                        + (1.0 - self.velocity_alpha)
                        * previous_velocity
                    )

                    # Update linear velocity
                    self.x_[9:12] = filtered_velocity

                else:

                    # Reset velocity after timing interruption
                    self.x_[9:12] = 0.0

            # Store current measurements
            self.previous_position = current_position
            self.previous_time = current_time

            # Update position
            self.x_[3:6] = current_position

            self.pose_received = True

            break

    def timer_callback(self):

        # Wait until both IMU and pose data are available
        if not self.imu_received or not self.pose_received:
            return

        msg = Float64MultiArray()

        msg.data = self.x_.tolist()

        self.state_publisher.publish(msg)


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
