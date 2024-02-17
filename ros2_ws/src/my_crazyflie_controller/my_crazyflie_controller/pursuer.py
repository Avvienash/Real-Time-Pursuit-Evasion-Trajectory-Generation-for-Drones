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


class PursuerNode(Node):
    
    def __init__(self, swarm):
        super().__init__("pursuer")
        self.get_logger().info("Pursuer Node has started")
                
        self.pursuer_traj = None
        self.pursuer_traj_temp = None
        self.pursuer_traj_received = False
        self.pursuer_traj_time = 0
        self.pursuer_traj_delay = 0.5
                
        #Pursuer traj Subscriber
        self.pursuer_traj_subscriber = self.create_subscription(
                                        Float64MultiArray,
                                        "/pursuer_traj",
                                        self.pursuer_traj_callback,
                                        10)
        
        # Game State Subscriber
        self.game_state_subscriber = self.create_subscription(
                                        Int32,
                                        "/game_state",
                                        self.game_state_callback,
                                        10)
        
        
        self.node_state = "getting_started"
        
        # create swarm
        self.swarm = swarm
        self.timeHelper = self.swarm.timeHelper
        self.h = 0.4 # target Height
        self.get_logger().info("Swarm Detected")
        
        
        # Get Pursuer
        pursuer_cf_list = [obj for obj in self.swarm.allcfs.crazyflies if getattr(obj, 'prefix', None) == "/cf2"]
        if pursuer_cf_list:
            print("CF2 Pursuer Found")
            self.pursuer_cf = pursuer_cf_list[0] 
        else:
            self.get_logger().warn('Pursuer not found')
            self.get_logger().info("Stopping node...")
            self.node_state = "stopped"
        
        self.node_state = "taking_off"
        self.pursuer_cf.takeoff(targetHeight=self.h, duration=4)
        time.sleep(4.5)
        self.node_state = "running_traj"
        
        # Create a controller callback
        self.create_timer(0.04,self.controller_callback)
        
    def game_state_callback(self, msg: Int32):
        if msg.data == 1 and self.node_state != "done":
            self.get_logger().info("Evader Caught, Stopping Pursuer")
            self.node_state = "stopped"
        elif msg.data == 2 and self.node_state != "done":
            self.get_logger().warn("Out of Bounds, Stopping Pursuer")
            self.node_state = "stopped"
            
    def pursuer_traj_callback(self, msg: Float64MultiArray):
        if msg.data:
            self.pursuer_traj = msg.data[:-1]
            self.pursuer_traj = np.array(self.pursuer_traj).reshape(9,6)
            self.pursuer_traj_delay = msg.data[-1]
            self.pursuer_traj_received = True
            self.pursuer_traj_time = self.timeHelper.time()
            
    def controller_callback(self):
        
        if self.node_state == "done":
            return
        
        if self.pursuer_traj_received:
            self.pursuer_traj_temp = self.pursuer_traj
            self.pursuer_traj_received = False
            self.get_logger().info("Trajectory Received and Updated")
        
        
        self.get_logger().info("State: " + self.node_state)
        if self.node_state == "running_traj" and (self.pursuer_traj_temp is not None):
            
            t = self.timeHelper.time() - self.pursuer_traj_time + self.pursuer_traj_delay
            
            if t > 9*0.2 :
                
                self.pursuer_cf.notifySetpointsStop()
                self.get_logger().info("Trajectory Complete, Waitinng for next Trajectory")
                
                self.pursuer_cf.notifySetpointsStop()
                self.pursuer_cf.takeoff(targetHeight=self.h, duration=0.2)
            
            else:
                i = np.floor(t/0.2).astype(int)
                self.pursuer_cf.cmdFullState(
                    np.array([self.pursuer_traj_temp[i,0], self.pursuer_traj_temp[i,1], self.h]),
                    np.array([self.pursuer_traj_temp[i,2], self.pursuer_traj_temp[i,3], 0.0]),
                    np.array([self.pursuer_traj_temp[i,4], self.pursuer_traj_temp[i,5], 0.0]),
                    0.0,
                    np.array([0.0,0.0,0.0]))
    
        elif self.node_state == "stopped":
            
            self.pursuer_cf.notifySetpointsStop()
            
            # Stay and blink
            self.pursuer_cf.takeoff(targetHeight=self.h, duration=0.2)
            for i in range(5):
                self.pursuer_cf.setParam('led.bitmask', 128)
                time.sleep(2)
                self.pursuer_cf.setParam('led.bitmask', 0)
            
            pos = self.pursuer_cf.initialPosition + np.array([0, 0, self.h])
            self.pursuer_cf.goTo(pos, 0, 4)
            time.sleep(4.5)
            
            self.pursuer_cf.land(targetHeight=0.03, duration=4)
            time.sleep(4.2)
            
            self.get_logger().info("Stop Command")
            self.node_state = "done"
        
        

            
            
            
def main(args=None):
    
    rclpy.init(args=args)
    swarm = Crazyswarm()
    
    try:
        node = PursuerNode(swarm)
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        pass
    
    try:
        print("Stopping Node")
        for cf in swarm.allcfs.crazyflies:
            cf.notifySetpointsStop()
        swarm.allcfs.land(targetHeight=0.03, duration=2)
        time.sleep(2.2)
    except Exception as e:
        print(e)

    if rclpy.ok():
        rclpy.shutdown()

    
if __name__ == '__main__':
    main()