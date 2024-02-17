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

from main import *
import uav_trajectory
from uav_trajectory import *


class TrajGenNode(Node):
    
    def __init__(self, generator):
        super().__init__("traj_gen")
        self.get_logger().info("Traj Gen Node has started")
        
        # Load Models
        self.generator = generator
        self.optimization_layer_pusuer = self.generator.contruct_optimization_problem(self.generator.pursuer_limits,'hard')
        self.optimization_layer_evader = self.generator.contruct_optimization_problem(self.generator.evader_limits,'hard')
        self.optimization_layer_pusuer_soft = self.generator.contruct_optimization_problem(self.generator.pursuer_limits,'soft')
        self.optimization_layer_evader_soft = self.generator.contruct_optimization_problem(self.generator.evader_limits,'soft')
        self.get_logger().info("Models Loaded")
        
        self.pursuer_state = torch.tensor([0.0,0.0,0.0,0.0]).to(self.generator.device)
        self.evader_state = torch.tensor([1.0,0.0,0.0,0.0]).to(self.generator.device)
        self.pursuer_traj = None
        self.evader_traj = None
        self.get_logger().info("States Initialized")
        
        #Pursuer State Subscriber
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
        
        self.puruser_traj_pub = self.create_publisher(Float64MultiArray, "/pursuer_traj", 10)
        self.evader_traj_pub = self.create_publisher(Float64MultiArray, "/evader_traj", 10)
    
        # create the trajectory generator
        self.create_timer(1.0,self.generate_trajectories)
 
    def pursuer_state_callback(self, msg: LogDataGeneric):
        if msg.values:
            self.pursuer_state = torch.tensor([msg.values[0],msg.values[1],msg.values[2],msg.values[3]]).to(self.generator.device)
         
    def evader_state_callback(self, msg: LogDataGeneric):
        if msg.values:
            self.evader_state = torch.tensor([msg.values[0],msg.values[1],msg.values[2],msg.values[3]]).to(self.generator.device)
                      
    def generate_trajectories(self):
        time_1 = datetime.now()
        
        with torch.no_grad():
            feasible_p, feasible_e,\
            pursuer_final_states, evader_final_states,\
            pursuer_final_controls, evader_final_controls,\
            pursuer_error, evader_error = self.generator.generate_trajectories(self.pursuer_state, self.evader_state, self.optimization_layer_pusuer, self.optimization_layer_evader)
        
        feasible = feasible_p and feasible_e
        if not feasible:
            self.get_logger().warn("States not Feasible, Trying Soft Constraints")
            
            with torch.no_grad():
                feasible_p, feasible_e,\
                pursuer_final_states, evader_final_states,\
                pursuer_final_controls, evader_final_controls,\
                pursuer_error, evader_error = self.generator.generate_trajectories(self.pursuer_state, self.evader_state, self.optimization_layer_pusuer_soft, self.optimization_layer_evader_soft)

            feasible = feasible_p and feasible_e
            if not feasible:
                self.get_logger().warn("States not Feasible, Stopping...")
                self.finish = True
                self.pursuer_traj = None
                self.evader_traj = None
                return
            
        
        pursuer_waypoints = np.append(pursuer_final_states[:-1,:].cpu().detach().numpy(),pursuer_final_controls[1:,:].cpu().detach().numpy(), axis=1)
        evader_waypoints = np.append(evader_final_states[:-1,:].cpu().detach().numpy(), evader_final_controls[1:,:].cpu().detach().numpy(), axis=1)
        
        delay = (datetime.now() - time_1).total_seconds()
        self.pursuer_traj = np.append(pursuer_waypoints.flatten(),delay).tolist()
        self.evader_traj = np.append(evader_waypoints.flatten(),delay).tolist()
        
        # for i in range(self.generator.n_steps):
        #     print("t: " , round(pursuer_waypoints[i,0],2), 
        #           "\t x: ", round(pursuer_waypoints[i,1],4), 
        #           "\t y: " ,round(pursuer_waypoints[i,2],4), 
        #           "\t z: " , round(pursuer_waypoints[i,3],2) , 
        #           "\t yaw: " , round(pursuer_waypoints[i,4],1) )
        
        # self.pursuer_traj = generate_trajectory(pursuer_waypoints, 1).to_msg()
        # self.evader_traj = generate_trajectory(evader_waypoints, 1).to_msg()
    
        msg = Float64MultiArray()
        msg.data = self.pursuer_traj
        self.puruser_traj_pub.publish(msg)
        
        msg = Float64MultiArray()
        msg.data = self.evader_traj
        self.evader_traj_pub.publish(msg)
        
        print("Delay: ", delay) 
        
        self.get_logger().info(" : %s" % ((datetime.now() - time_1).total_seconds()))

        
        
        
def main(args=None):
    
    # Enable Logging
    logging.basicConfig(level=logging.INFO, 
                        format='[%(levelname)s] %(message)s',
                        handlers=[ logging.StreamHandler()])
        
    load_version  = 1
    logging.info("Loading version: %s", load_version)
    params_path = 'training/train_v' + str(load_version) + '/model_params.txt'
    params_path =  Path("/mnt/c/Users/Avvienash/Documents/SRP/Trajectory Generation Code") / params_path
    params = {}
    
    with open(params_path, 'r') as f:
        for line in f:
            key, value = line.strip().split(': ')
            # Convert the value to int or float if possible
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    if value == 'True':
                        value = True
                    elif value == 'False':
                        value = False  # Keep the value as a string if conversion is not possible

            params[key] = value
            
    # Now, you can access the parameters using loaded_params dictionary
    logging.info("Loaded Parameters: %s", params)
    
    generator = PlayerTrajectoryGenerator(num_traj = params['num_traj'],
                                          state_dim = params['state_dim'],
                                          input_dim = params['input_dim'],
                                          n_steps = params['n_steps'],
                                          dt = params['dt'],
                                          pursuer_limits = ast.literal_eval(params['pursuer_limits']),
                                          evader_limits = ast.literal_eval(params['evader_limits']),
                                          catch_radius = params['catch_radius'],
                                          hidden_layer_num = params['hidden_layer_num'],
                                          solver_max_iter = params['solver_max_iter'],
                                          save_path = " ",
                                          device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                          verbose = params['verbose'],
                                          solve_method = params['solve_method'],
                                          enviroment= params['enviroment'],
                                          margin= params['margin'],
                                          bounds_type= params['bounds_type'])
    
                                 
     # Load the model
    pursuer_path = 'training/train_v' + str(load_version) + '/pursuer_model.pth'
    pursuer_path =  Path("/mnt/c/Users/Avvienash/Documents/SRP/Trajectory Generation Code") / pursuer_path
    generator.pursuer_model.load_state_dict(torch.load(pursuer_path))
    
    evader_path = 'training/train_v' + str(load_version) + '/evader_model.pth'
    evader_path =  Path("/mnt/c/Users/Avvienash/Documents/SRP/Trajectory Generation Code") / evader_path
    generator.evader_model.load_state_dict(torch.load(evader_path))
    
    
    
    rclpy.init(args=args)
    node = TrajGenNode(generator)
    rclpy.spin(node)
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()