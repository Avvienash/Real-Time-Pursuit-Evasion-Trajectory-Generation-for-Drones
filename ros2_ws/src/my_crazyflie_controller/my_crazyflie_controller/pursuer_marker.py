#! /usr/bin/env python3

import rclpy
from rclpy.node import Node
from motion_capture_tracking_interfaces.msg import NamedPoseArray
from crazyflie_interfaces.msg import LogDataGeneric
from crazyflie_py import genericJoystick
from crazyflie_py.uav_trajectory import Trajectory
from crazyflie_py import Crazyswarm
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Int32
from datetime import datetime


class PursuerNode(Node):
    
    def __init__(self):
        super().__init__("pursuer")
        self.get_logger().info("Pursuer Node has started")
            
        # create publisher for state
        self.pursuer_state_publisher = self.create_publisher(LogDataGeneric, "/cf2/state", 10)
            
        # Create a controller callback
        
        self.pursuer_pose_subscriber = self.create_subscription(
                                        NamedPoseArray,
                                        "/poses",
                                        self.pursuer_pose_callback,
                                        rclpy.qos.qos_profile_sensor_data)
        self.state = [0.0, 0.0, 0.0, 0.0]
        
        self.t = datetime.now()
        
        
        

    def pursuer_pose_callback(self, msg: NamedPoseArray):
        if msg.poses:
            for named_pose in msg.poses:
                name = named_pose.name
                if name == "cf2":
                    pose = named_pose.pose.position
                    
                    dt = float((datetime.now()-self.t).total_seconds())
                    self.state = [float(pose.x), float(pose.y), float((pose.x-self.state[0])/dt), float((pose.y-self.state[1])/dt)]
                    self.t = datetime.now()
                    
                    msg = LogDataGeneric()
                    msg.values = self.state
                    self.pursuer_state_publisher.publish(msg)
                    
                    self.get_logger().info("Pursuer State: " + str(self.state))
                    return
        
    
            
def main(args=None):
    
    rclpy.init(args=args)
    node = PursuerNode()
    rclpy.spin(node)
    rclpy.shutdown()

    
if __name__ == '__main__':
    main()