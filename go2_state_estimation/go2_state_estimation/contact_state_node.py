import time

import rclpy
from rclpy.node import Node

from ros_gz_interfaces.msg import Contacts
from std_msgs.msg import Float64MultiArray


class ContactStateNode(Node):

    def __init__(self):
        super().__init__('contact_state_node')

        self.legs = ['LF', 'RF', 'LR', 'RR']

        self.contact_state = {leg: 0 for leg in self.legs}
        self.last_message_time = {leg: None for leg in self.legs}

        self.timeout_seconds = 0.2

        self.publisher = self.create_publisher(
            Float64MultiArray,
            '/leg_stance_state',
            10
        )

        for leg in self.legs:
            topic = (
                f'/world/empty/model/go2_robot/link/'
                f'{leg}_calf/sensor/{leg}_contact/contact'
            )

            self.create_subscription(
                Contacts,
                topic,
                lambda msg, leg=leg: self.contact_callback(msg, leg),
                10
            )

            self.get_logger().info(f'Subscribed to {topic}')

        self.timer = self.create_timer(
            0.02,
            self.publish_stance_state
        )

        self.get_logger().info(
            'Contact state node started. Order: [LF, RF, LR, RR]'
        )

    def contact_callback(self, msg, leg):
        foot_name = f'{leg}_foot_collision'

        ground_contact = False

        for contact in msg.contacts:
            name1 = contact.collision1.name
            name2 = contact.collision2.name

            foot_ground_pair = (
                (foot_name in name1 and
                 'ground_plane::link::collision' in name2)
                or
                (foot_name in name2 and
                 'ground_plane::link::collision' in name1)
            )

            if foot_ground_pair:
                ground_contact = True
                break

        self.contact_state[leg] = int(ground_contact)
        self.last_message_time[leg] = time.monotonic()

    def publish_stance_state(self):
        now = time.monotonic()

        values = []

        for leg in self.legs:
            last_time = self.last_message_time[leg]

            if (
                last_time is None
                or now - last_time > self.timeout_seconds
            ):
                values.append(0)
            else:
                values.append(self.contact_state[leg])

        msg = Float64MultiArray()
        msg.data = [float(value) for value in values]

        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    node = ContactStateNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
