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
    
    def __init__(self, swarm):
        super().__init__("evader")
        self.get_logger().info("Evader Node has started")
                
        self.evader_traj = None
        self.evader_traj_temp = None
        self.evader_traj_received = False
        self.evader_traj_time = 0
        self.evader_traj_delay = 0.1
                
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
        
        # create swarm
        self.swarm = swarm
        self.timeHelper = self.swarm.timeHelper
        self.h = 0.2 # target Height
        self.get_logger().info("Swarm Detected")
        
        
        # Get Evader
        evader_cf_list = [obj for obj in self.swarm.allcfs.crazyflies if getattr(obj, 'prefix', None) == "/cf5"]
        if evader_cf_list:
            print("CF5 Evader Found")
            self.evader_cf = evader_cf_list[0] 
        else:
            self.get_logger().warn('Evader not found')
            self.get_logger().info("Stopping node...")
            self.node_state = "stopped"
        
        self.node_state = "taking_off"
        self.evader_cf.takeoff(targetHeight=self.h, duration=4)
        time.sleep(4.5)
        self.node_state = "running_traj"
        
        # Create a controller callback
        self.create_timer(0.04,self.controller_callback)
        
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
            self.evader_traj_time = self.timeHelper.time()
            
    def controller_callback(self):
        
        if self.node_state == "done":
            return
        
        if self.evader_traj_received:
            self.evader_traj_temp = self.evader_traj
            self.evader_traj_received = False
            self.get_logger().info("Trajectory Received and Updated")
        
        
        self.get_logger().info("State: " + self.node_state)
        if self.node_state == "running_traj" and (self.evader_traj_temp is not None):
            
            t = self.timeHelper.time() - self.evader_traj_time + self.evader_traj_delay
            
            if t > 9*0.2 :
                
                self.evader_cf.notifySetpointsStop()
                self.get_logger().info("Trajectory Complete, Waitinng for next Trajectory")
                
                self.evader_cf.notifySetpointsStop()
                self.evader_cf.takeoff(targetHeight=self.h, duration=0.2)
            
            else:
                i = np.floor(t/0.2).astype(int)
                self.evader_cf.cmdFullState(
                    np.array([self.evader_traj_temp[i,0], self.evader_traj_temp[i,1], self.h]),
                    np.array([self.evader_traj_temp[i,2], self.evader_traj_temp[i,3], 0.0]),
                    np.array([self.evader_traj_temp[i,4], self.evader_traj_temp[i,5], 0.0]),
                    0.0,
                    np.array([0.0,0.0,0.0]))
    
        elif self.node_state == "stopped":
            
            self.evader_cf.notifySetpointsStop()
            
            # Stay and blink
            self.evader_cf.takeoff(targetHeight=self.h, duration=0.2)
            for i in range(5):
                self.evader_cf.setParam('led.bitmask', 128)
                time.sleep(2)
                self.evader_cf.setParam('led.bitmask', 0)
                
            pos = self.evader_cf.initialPosition + np.array([0, 0, self.h])
            self.evader_cf.goTo(pos, 0, 4)
            time.sleep(5)
            self.evader_cf.land(targetHeight=0.03, duration=4)
            time.sleep(4.2)
            
            self.get_logger().info("Stop Command")
            self.node_state = "done"
        
        

            
            
            
def main(args=None):
    
    rclpy.init(args=args)
    swarm = Crazyswarm()
    
    try:
        node = EvaderNode(swarm)
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