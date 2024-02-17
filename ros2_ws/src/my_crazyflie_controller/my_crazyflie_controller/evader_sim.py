#! /usr/bin/env python3
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))
windows_path = Path("/mnt/c/Users/Avvienash/Documents/SRP/Trajectory Generation Code")
sys.path.append(str(Path("/mnt/c/Users/Avvienash/Documents/SRP/Trajectory Generation Code")))

import math
import os
import random
import numpy as np
import time

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import cvxpy as cp
import nashpy as nash
import logging
import ast

from cvxpylayers.torch import CvxpyLayer
from cvxpy.problems.objective import Maximize, Minimize
from matplotlib.animation import FFMpegWriter
from matplotlib import cm
from IPython.display import Video
from datetime import datetime


import argparse
import scipy.optimize

import rclpy
from rclpy.node import Node
from motion_capture_tracking_interfaces.msg import NamedPoseArray
from crazyflie_interfaces.msg import LogDataGeneric
from crazyflie_py import genericJoystick
from crazyflie_py.uav_trajectory import Trajectory
from crazyflie_py import Crazyswarm
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Int32

from main import *
import uav_trajectory
from uav_trajectory import *


class EvaderNode(Node):
    
    def __init__(self):
        super().__init__("evader")
        self.get_logger().info("Evader Node has started")
                
        self.evader_traj = None
        self.evader_traj_temp = None
        self.evader_traj_received = False
        self.evader_traj_time = 0
        self.evader_traj_delay = 0.5
                
        #Evader traj Subscriber
        self.evader_traj_subscriber = self.create_subscription(
                                        Float64MultiArray,
                                        "/evader_traj",
                                        self.evader_traj_callback,
                                        10)
        
        # Game State Subscriber
        self.game_state_subscriber = self.create_subscription(
                                        Int32,
                                        "/game_state",
                                        self.game_state_callback,
                                        10)
        
        
        self.node_state = "getting_started"
        
        # create publisher for state
        self.evader_state_publisher = self.create_publisher(LogDataGeneric, "/cf5/state", 10)
        
        self.node_state = "running_traj"
        
        
        # Create a controller callback
        self.state = [1.0, 0, 0, 0]
        self.create_timer(0.1,self.controller_callback)
        
        
    def game_state_callback(self, msg: Int32):
        if msg.data == 1 and self.node_state != "done":
            self.get_logger().info("Evader Caught, Stopping Evader")
            self.node_state = "stopped"
        elif msg.data == 2 and self.node_state != "done":
            self.get_logger().warn("Out of Bounds, Stopping Evader")
            self.node_state = "stopped"
            
    def evader_traj_callback(self, msg: Float64MultiArray):
        if msg.data:
            self.evader_traj = msg.data[:-1]
            self.evader_traj = np.array(self.evader_traj).reshape(9,6)
            self.evader_traj_delay = msg.data[-1]
            self.evader_traj_received = True
            self.evader_traj_time = datetime.now()
            
    def controller_callback(self):
        
        if self.node_state == "done":
            return
        
        if self.evader_traj_received:
            self.evader_traj_temp = self.evader_traj
            self.evader_traj_received = False
            self.get_logger().info("Trajectory Received and Updated")
        
        
        self.get_logger().info("State: " + self.node_state)
        if self.node_state == "running_traj" and (self.evader_traj_temp is not None):
            t = (datetime.now() - self.evader_traj_time).total_seconds() + self.evader_traj_delay
            
            if t > 9*0.2:
                
                self.get_logger().info("Trajectory Finished")
                
            else:
                i = np.floor(t/0.2).astype(int)
                self.state = self.evader_traj_temp[i,:4]
    
        elif self.node_state == "stopped":
                        
            self.get_logger().info("Stop Command")
            self.node_state = "done"

        
        msg = LogDataGeneric()
        msg.values = [float(self.state[0]), float(self.state[1]), float(self.state[2]), float(self.state[3])]
        self.evader_state_publisher.publish(msg)
    
            
def main(args=None):
    
    rclpy.init(args=args)
    node = EvaderNode()
    rclpy.spin(node)
    rclpy.shutdown()

    
if __name__ == '__main__':
    main()