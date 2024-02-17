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

import threading
lock = threading.Lock()

class PlotterNode(Node):
    
    def __init__(self):
        super().__init__("plotter")
        self.get_logger().info("Plotter Node has started")
                
        self.pursuer_state = np.array([0.0,0.0,0.0,0.0])
        self.evader_state = np.array([1.0,0.0,0.0,0.0])
        self.pursuer_traj = None
        self.evader_traj = None
        
        
        
        #Pursuer state Subscriber
        self.pursuer_state_subscriber = self.create_subscription(
                                        LogDataGeneric,
                                        "/cf2/state",
                                        self.pursuer_state_callback,
                                        10)
        
        self.evader_state_subscriber = self.create_subscription(
                                        LogDataGeneric,
                                        "/cf5/state",
                                        self.evader_state_callback,
                                        10)
        
        
        #Pursuer traj Subscriber
        self.pursuer_traj_subscriber = self.create_subscription(
                                        Float64MultiArray,
                                        "/pursuer_traj",
                                        self.pursuer_traj_callback,
                                        10)
        
        # Evader traj Subscriber
        self.evader_traj_subscriber = self.create_subscription(
                                        Float64MultiArray,
                                        "/evader_traj",
                                        self.evader_traj_callback,
                                        10)

        
        # Create Trajectory Plotter:
        self.fig, self.ax = plt.subplots()
        # plot the border
        r  = 2
        border_x = r*np.sin(np.arange(0, 2*np.pi + 2*np.pi/5, 2*np.pi/5))
        border_y = r*np.cos(np.arange(0, 2*np.pi + 2*np.pi/5, 2*np.pi/5))
        self.ax.plot(border_x, border_y, 'k-', linewidth=1)
        self.ax.set_xlabel('x position')
        self.ax.set_ylabel('y position')
        self.ax.set_title('Top Down Simulation')
        self.ax.set_xlim(-(r+0.2), (r+0.2))
        self.ax.set_ylim(-(r+0.2), (r+0.2))
        
        # Left subplot: Top-down simulation
        self.pursuer_pos_plot, = self.ax.plot([], [], 'ro', markersize=8)
        self.pursuer_traj_plot, = self.ax.plot([], [], 'r-')
        self.evader_pos_plot, = self.ax.plot([], [], 'bo', markersize=8)
        self.evader_traj_plot, = self.ax.plot([], [], 'b-')
        
        # Create a Plotter
        self.create_timer(0.3,self.plotter)
        
        #Create a game state publisher
        self.game_state = 0
        self.game_state_publisher = self.create_publisher(Int32, "/game_state", 10)
        self.create_timer(0.1,self.game_state_publisher_callback)
    
    def game_state_publisher_callback(self):
        if np.linalg.norm(self.pursuer_state[:2] - self.evader_state[:2]) < 0.1:
            msg = Int32()
            msg.data = 1
            self.game_state_publisher.publish(msg)
            self.game_state = 1
        
        if np.any(np.abs(self.pursuer_state[:2]) > 2.1) or np.any(np.abs(self.evader_state[:2]) > 2.1):
            msg = Int32()
            msg.data = 2
            self.game_state_publisher.publish(msg)
            self.game_state = 2
        
    def pursuer_state_callback(self, msg: LogDataGeneric):
        if msg.values:
            self.pursuer_state = np.array([msg.values[0],msg.values[1],msg.values[2],msg.values[3]])
            
    def evader_state_callback(self, msg: LogDataGeneric):
        if msg.values:
            self.evader_state = np.array([msg.values[0],msg.values[1],msg.values[2],msg.values[3]])
            
    def pursuer_traj_callback(self, msg: Float64MultiArray):
        if msg.data:
            #self.pursuer_traj = Trajectory()
            #self.pursuer_traj.from_msg(msg.data)
            self.pursuer_traj = msg.data[:-1]
            self.pursuer_traj = np.array(self.pursuer_traj).reshape(9,6)
    
    def evader_traj_callback(self, msg: Float64MultiArray):
        if msg.data:
            #self.evader_traj = Trajectory()
            #self.evader_traj.from_msg(msg.data)
            self.evader_traj = msg.data[:-1]
            self.evader_traj = np.array(self.evader_traj).reshape(9,6)
            
             
    def plotter(self):
        time_1 = datetime.now()
        
        if self.game_state == 1:
            self.ax.set_title('Evader Caught')
            plt.pause(0.01)
            return
        
        if self.pursuer_traj is not None:
            # ts = np.arange(0, self.pursuer_traj.duration, 0.01)
            # evals = np.empty((len(ts), 15))
            # for t, i in zip(ts, range(0, len(ts))):
            #     e = self.pursuer_traj.eval(t)
            #     evals[i, 0:3]  = e.pos
            #     evals[i, 3:6]  = e.vel
            #     evals[i, 6:9]  = e.acc
            #     evals[i, 9:12] = e.omega
            #     evals[i, 12]   = e.yaw
            #     evals[i, 13]   = e.roll
            #     evals[i, 14]   = e.pitch

            # velocity = np.linalg.norm(evals[:,3:6], axis=1)
            # acceleration = np.linalg.norm(evals[:,6:9], axis=1)
            # omega = np.linalg.norm(evals[:,9:12], axis=1)
            
            self.pursuer_traj_plot.set_data(self.pursuer_traj[:,0],self.pursuer_traj[:,1])
        else:
            self.pursuer_traj_plot.set_data([],[])
            
        if self.evader_traj is not None:
            # ts = np.arange(0, self.evader_traj.duration, 0.01)
            # evals = np.empty((len(ts), 15))
            # for t, i in zip(ts, range(0, len(ts))):
            #     e = self.evader_traj.eval(t)
            #     evals[i, 0:3]  = e.pos
            #     evals[i, 3:6]  = e.vel
            #     evals[i, 6:9]  = e.acc
            #     evals[i, 9:12] = e.omega
            #     evals[i, 12]   = e.yaw
            #     evals[i, 13]   = e.roll
            #     evals[i, 14]   = e.pitch

            # velocity = np.linalg.norm(evals[:,3:6], axis=1)
            # acceleration = np.linalg.norm(evals[:,6:9], axis=1)
            # omega = np.linalg.norm(evals[:,9:12], axis=1)
            
            self.evader_traj_plot.set_data(self.evader_traj[:,0],self.evader_traj[:,1])
        else:
            self.evader_traj_plot.set_data([],[])
            
            
        self.pursuer_pos_plot.set_data(self.pursuer_state[0],self.pursuer_state[1])
        self.evader_pos_plot.set_data(self.evader_state[0],self.evader_state[1])
        plt.pause(0.01)
        self.get_logger().info("delay: %s" % (datetime.now() - time_1).total_seconds())

def main(args=None):
    
    rclpy.init(args=args)
    node = PlotterNode()
    rclpy.spin(node)
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()