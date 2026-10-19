#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float64MultiArray


class FootPositionNode(Node):

    def __init__(self):
        super().__init__('foot_position_node')

        # Go2 geometry from go2_description/xacro/const.xacro
        self.thigh_length = 0.213
        self.calf_length = 0.213
        self.thigh_offset = 0.0955

        self.hip_offset_x = 0.1934
        self.hip_offset_y = 0.0465

        # Leg order required by MPC
        self.legs = ['LF', 'RF', 'LR', 'RR']

        self.joint_positions = {}
        self.orientation = None

        self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )

        self.create_subscription(
            Imu,
            '/trunk_imu',
            self.imu_callback,
            10
        )

        self.publisher = self.create_publisher(
            Float64MultiArray,
            '/foot_positions',
            10
        )

        self.create_timer(0.02, self.timer_callback)

        self.get_logger().info(
            'Foot-position publisher started'
        )

    def joint_callback(self, msg):
        for name, position in zip(msg.name, msg.position):
            self.joint_positions[name] = position

    def imu_callback(self, msg):
        q = msg.orientation

        norm = math.sqrt(
            q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w
        )

        if norm < 1e-8:
            return

        self.orientation = (
            q.x / norm,
            q.y / norm,
            q.z / norm,
            q.w / norm
        )

    def rotate_to_world(self, vector):
        """
        Rotate a vector from the body frame to the world frame
        using the IMU quaternion.
        """

        x, y, z, w = self.orientation

        # Quaternion rotation matrix
        R = [
            [
                1 - 2*(y*y + z*z),
                2*(x*y - z*w),
                2*(x*z + y*w)
            ],
            [
                2*(x*y + z*w),
                1 - 2*(x*x + z*z),
                2*(y*z - x*w)
            ],
            [
                2*(x*z - y*w),
                2*(y*z + x*w),
                1 - 2*(x*x + y*y)
            ]
        ]

        return [
            sum(R[i][j] * vector[j] for j in range(3))
            for i in range(3)
        ]

    def calculate_foot_position(self, leg, q1, q2, q3):

        mirror = 1.0 if leg in ('LF', 'LR') else -1.0

        front = 1.0 if leg in ('LF', 'RF') else -1.0

        L2 = self.thigh_length
        L3 = self.calf_length
        d = self.thigh_offset

        A = (
            L2 * math.sin(q2)
            + L3 * math.sin(q2 + q3)
        )

        B = (
            L2 * math.cos(q2)
            + L3 * math.cos(q2 + q3)
        )

        # Foot relative to hip
        x = -A

        y = (
            mirror * d * math.cos(q1)
            + B * math.sin(q1)
        )

        z = (
            mirror * d * math.sin(q1)
            - B * math.cos(q1)
        )

        # Hip position relative to trunk
        x += front * self.hip_offset_x
        y += mirror * self.hip_offset_y

        return [x, y, z]

    def timer_callback(self):

        if self.orientation is None:
            return

        positions = []

        for leg in self.legs:

            names = [
                f'{leg}_hip_joint',
                f'{leg}_upper_leg_joint',
                f'{leg}_lower_leg_joint'
            ]

            if not all(
                name in self.joint_positions
                for name in names
            ):
                return

            q1, q2, q3 = [
                self.joint_positions[name]
                for name in names
            ]

            foot_body = self.calculate_foot_position(
                leg, q1, q2, q3
            )

            # Approximation:
            # trunk origin is used as the COM reference.
            #
            # We will replace this with the actual
            # whole-body COM offset later.

            foot_world = self.rotate_to_world(foot_body)

            positions.extend(foot_world)

        msg = Float64MultiArray()
        msg.data = positions

        self.publisher.publish(msg)


def main(args=None):

    rclpy.init(args=args)

    node = FootPositionNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
