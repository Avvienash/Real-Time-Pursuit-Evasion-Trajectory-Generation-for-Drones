""" 
Author: Avvienash
Date: 12/1/2024
Description:
    This file is used to test the trained networks.
    The test is done by running a simulation between the pursuer and the evader.
"""

""" Importing Libraries """
#region Importing Libraries
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import cvxpy as cp
import nashpy as nash

from cvxpylayers.torch import CvxpyLayer
from matplotlib.animation import FFMpegWriter
from IPython.display import Video
from datetime import datetime
import os

from utils import *

plt.rcParams['animation.ffmpeg_path'] = 'C:\\Users\\Avvienash\\Documents\\ffmpeg-6.1-essentials_build\\ffmpeg-6.1-essentials_build\\bin\\ffmpeg.exe'



class Controller:
    def __init__(self, version):
        
        self.version = version
        print("Using Version:", self.version)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # device
        print("Using Device:", self.device)
        
        # Load the trained networks
        print("---------------------------------------------------------------")
        print("Loading the Model Parameters")
        print("---------------------------------------------------------------")
        
        self.loaded_params = {}
        file_path = f'weights/model_params_v{self.version}.txt'
        print("File Path:", file_path)
        
        with open(file_path, 'r') as f:
            for line in f:
                key, value = line.strip().split(': ')
                # Convert the value to int or float if possible
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass  # Keep the value as a string if conversion is not possible

                self.loaded_params[key] = value
                
        # Now, you can access the parameters using loaded_params dictionary
        print("Loaded Parameters:", self.loaded_params)
        
        
        
        # Load the trained networks
        print("---------------------------------------------------------------")
        print("Loading the trained networks")
        print("---------------------------------------------------------------")
        
        # Load the trained networks
        self.pursuer_net = PlayerTrajectoryGenerator(
                            self.loaded_params["pursuer_num_traj"], 
                            self.loaded_params["dim_x"], 
                            self.loaded_params["n_steps"], 
                            self.loaded_params["xy_limit"], 
                            self.loaded_params["acc_limit"],
                            self.loaded_params["input_layer_num"], 
                            self.loaded_params["hidden_layer_num"], 
                            self.loaded_params["output_layer_num"], 
                            self.device).to(self.device)
        
        self.evader_net = PlayerTrajectoryGenerator(
                            self.loaded_params["evader_num_traj"], 
                            self.loaded_params["dim_x"], 
                            self.loaded_params["n_steps"], 
                            self.loaded_params["xy_limit"], 
                            self.loaded_params["acc_limit"],
                            self.loaded_params["input_layer_num"], 
                            self.loaded_params["hidden_layer_num"], 
                            self.loaded_params["output_layer_num"], 
                            self.device).to(self.device)
        
        self.pursuer_net.load_state_dict(torch.load(f'weights/pursuer_weights_v{version}.pth'))
        self.evader_net.load_state_dict(torch.load(f'weights/evader_weights_v{version}.pth'))
        
        print("Successfully loaded the trained networks")
        
        # Create trajectory generator
        print("---------------------------------------------------------------")
        print("Creating the Trajectory Generator")
        print("---------------------------------------------------------------")
        

        # Create trajectory generator
        self.traj_generator = construct_mpc_problem(
                            self.loaded_params["dim_x"], 
                            self.loaded_params["dim_u"], 
                            self.loaded_params["n_steps"], 
                            self.loaded_params["xy_limit"], 
                            self.loaded_params["acc_limit"], 
                            self.loaded_params["dt"], 
                            self.loaded_params["W_state"], 
                            self.loaded_params["W_control"],
                            self.loaded_params["output_layer_num"])
        
        self.MSE_loss = nn.MSELoss()

        # Set networks to evaluation mode
        self.pursuer_net.eval()
        self.evader_net.eval()

    def step(self, pursuer_input, evader_input):
        
        with torch.no_grad():
            evader_output = self.evader_net(evader_input)
            pursuer_output = self.pursuer_net(pursuer_input)

        pursuer_traj = GetTrajFromBatchinput(
                            pursuer_output, 
                            pursuer_input, 
                            self.loaded_params["pursuer_num_traj"],
                            self.traj_generator, 
                            self.loaded_params["solver_max_iter"], 
                            self.device)
        
        pursuer_traj_ref = pursuer_traj.clone().detach()

        evader_traj = GetTrajFromBatchinput(
                            evader_output, 
                            evader_input, 
                            self.loaded_params["evader_num_traj"],
                            self.traj_generator,
                            self.loaded_params["solver_max_iter"],
                            self.device)
    
        
        evader_traj_ref = evader_traj.clone().detach()

        # Create the Bimatrix Game for the Pursuer
        pursuer_BMG_matrix = torch.zeros((self.loaded_params["pursuer_num_traj"], self.loaded_params["evader_num_traj"]))
        for i in range(self.loaded_params["pursuer_num_traj"]):
            for j in range(self.loaded_params["evader_num_traj"]):
                pursuer_BMG_matrix[i][j] = self.MSE_loss(pursuer_traj[i], evader_traj_ref[j])

        # Create the Bimatrix Game for the Evader
        evader_BMG_matrix = torch.zeros((self.loaded_params["evader_num_traj"], self.loaded_params["pursuer_num_traj"]))
        for i in range(self.loaded_params["evader_num_traj"]):
            for j in range(self.loaded_params["pursuer_num_traj"]):
                evader_BMG_matrix[i][j] = -1 * self.MSE_loss(evader_traj[i], pursuer_traj_ref[j])

        # Solve the Bimatrix Game
        pursuer_BMG_matrix_np = pursuer_BMG_matrix.clone().detach().numpy()
        evader_BMG_matrix_np = evader_BMG_matrix.clone().detach().numpy()
        
        game = nash.Game(pursuer_BMG_matrix_np, evader_BMG_matrix_np)
        equilibria = game.lemke_howson_enumeration()

        sorted_equilibria = sorted(equilibria, key=lambda x: sum(x[0] * pursuer_BMG_matrix_np @ x[1]))
        pursuer_sol = torch.tensor(sorted_equilibria[0][0], dtype=torch.float)
        evader_sol = torch.tensor(sorted_equilibria[0][1], dtype=torch.float)

        # Calculate the trajectory
        pursuer_final_traj = torch.mm(pursuer_sol.view(1, -1).to(self.device), pursuer_traj)
        evader_final_traj = torch.mm(evader_sol.view(1, -1).to(self.device), evader_traj)
        pursuer_final_traj = pursuer_final_traj.squeeze()
        evader_final_traj = evader_final_traj.squeeze()

        return pursuer_final_traj, evader_final_traj

    
    def full_sim(self, pursuer_input, evader_input, max_steps=100):
        
        
        # Load the trained networks
        print("---------------------------------------------------------------")
        print("Loading the Model Parameters")
        print("---------------------------------------------------------------")
        
        pursuer_states = np.zeros((max_steps, 4))
        evader_states = np.zeros((max_steps, 4))
        
        pursuer_trajectories = np.zeros((max_steps, self.loaded_params["n_steps"]*self.loaded_params["dim_x"]))
        evader_trajectories = np.zeros((max_steps, self.loaded_params["n_steps"]*self.loaded_params["dim_x"]))
        
        for frame in range(max_steps):
            
            pursuer_final_traj, evader_final_traj = self.step(pursuer_input, evader_input)
            # Store the states in the array
            pursuer_states[frame,:] = pursuer_input.cpu().clone().detach().numpy()[:4]
            evader_states[frame,:] = evader_input.cpu().clone().detach().numpy()[:4]
            pursuer_trajectories[frame,:] = pursuer_final_traj.cpu().clone().detach().numpy()  
            evader_trajectories[frame,:] = evader_final_traj.cpu().clone().detach().numpy()
            
            # update the states
            pursuer_input = torch.tensor([*pursuer_final_traj.clone().detach()[:4],*evader_final_traj.clone().detach()[:4]], dtype=torch.float)
            evader_input = torch.tensor([*evader_final_traj.clone().detach()[:4],*pursuer_final_traj.clone().detach()[:4]], dtype=torch.float)
            
            if (abs(evader_final_traj[0] - pursuer_final_traj[0]) <= 0.2) and  (abs(evader_final_traj[1] - pursuer_final_traj[1]) <= 0.2) :
                pursuer_states = pursuer_states[:frame+1]
                evader_states = evader_states[:frame+1]
                pursuer_trajectories = pursuer_trajectories[:frame+1]
                evader_trajectories = evader_trajectories[:frame+1]
                print("Evader Caught")
                break
            
            print("Frame:", frame)
            
        return pursuer_states, evader_states, pursuer_trajectories, evader_trajectories
    

def main():
    
    # Create the controller
    version = get_latest_version('weights',"pursuer_weights_v")
    controller = Controller(version=version)
    
    # Set the initial states
    pursuer_input = torch.tensor([0, -2, 0, 0, 0, 0, 0, 0], dtype=torch.float)
    evader_input = torch.tensor([4, 4, 0, 0, 0, 0, 0, 0], dtype=torch.float)
    
    # Run the simulation
    pursuer_states, evader_states, pursuer_trajectories, evader_trajectories = controller.full_sim(pursuer_input, evader_input, max_steps=200)
    
    print("Evader States Shape: ", evader_states.shape)
    print("Pursuer States Shape: ", pursuer_states.shape)
    print("Evader Trajectories Shape: ", evader_trajectories.shape)
    print("Pursuer Trajectories Shape: ", pursuer_trajectories.shape)
    
    print("----------------------------------------------")
    print("Running Animation")
    print("----------------------------------------------")
    
    video_version = get_latest_version('videos',"demo_testing_v")
    
    # Animate the simulation
    animate(fps = 10, 
            name = "videos/demo_testing_v4.mp4", 
            pursuer_states = pursuer_states, 
            evader_states = evader_states, 
            pursuer_trajectories = pursuer_trajectories, 
            evader_trajectories = evader_trajectories)
    
    print("Animation Complete")

if __name__ == "__main__":
    main()