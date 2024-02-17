#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt
import rclpy
from std_msgs.msg import Float64MultiArray

def test_print(msg):
    if msg.data:
        print(msg.data[0])

def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node("test_node")
    node.get_logger().info("Test Node has started")
    node.create_subscription(Float64MultiArray, "pursuer_traj", test_print, 10)
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
