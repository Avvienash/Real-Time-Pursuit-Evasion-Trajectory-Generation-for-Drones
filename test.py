









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

from utils import *
from train import *

plt.rcParams['animation.ffmpeg_path'] = 'C:\\Users\\Avvienash\\Documents\\ffmpeg-6.1-essentials_build\\ffmpeg-6.1-essentials_build\\bin\\ffmpeg.exe'
#endregion

""" Global Variables """

pursuer_init_state = np.array([-4,-4,0,0])
evader_init_state = np.array([4,4,0,0])
pursuer_weights_version = 'weights/pursuer_weights_v2.pth'
evader_weights_version = 'weights/evader_weights_v2.pth'
frames = 200 # number of frames in the simulation
fps = 15
name = 'videos/Trajectory_Game_v1.mp4'




""" Main Function """

def main():
    
    # Load the trained networks
    print("---------------------------------------------------------------")
    print("Loading the trained networks")
    print("---------------------------------------------------------------")
    
    # Create the neural networks
    pursuer_net = PlayerTrajectoryGenerator(pursuer_num_traj,dim_x,n_steps,xy_limit,acc_limit,input_layer_num,hidden_layer_num,output_layer_num,device).to(device) # create the pursuer network 
    evader_net = PlayerTrajectoryGenerator(evader_num_traj,dim_x,n_steps,xy_limit,acc_limit,input_layer_num,hidden_layer_num,output_layer_num,device).to(device)  # create the evader network
    pursuer_net.load_state_dict(torch.load(pursuer_weights_version)) # load the pursuer network
    evader_net.load_state_dict(torch.load(pursuer_weights_version)) # load the evader network
    print("Successfully loaded the trained networks")
    
    
    # Run Test
    print("---------------------------------------------------------------")
    print("Running Test")
    print("---------------------------------------------------------------")

    
    pursuer_input = torch.tensor(np.concatenate((pursuer_init_state, evader_init_state)),dtype =torch.float) # initial input for the pursuer
    evader_input  = torch.tensor(np.concatenate((evader_init_state, pursuer_init_state)),dtype =torch.float) # initial input for the evader
    pursuer_states = np.zeros((frames,dim_x)) # array to store the pursuer states
    evader_states = np.zeros((frames,dim_x)) # array to store the evader states
    pursuer_trajectories = np.zeros((frames,dim_x*n_steps)) # array to store the pursuer trajectories
    evader_trajectories = np.zeros((frames,dim_x*n_steps)) # array to store the evader trajectories



    pursuer_net.eval() # set the pursuer network to evaluation mode
    evader_net.eval() # set the evader network to evaluation mode
    
    traj_generator = construct_mpc_problem(dim_x,dim_u,n_steps,xy_limit,acc_limit,dt,W_state,W_control,output_layer_num)
    MSE_loss = nn.MSELoss() # create the loss function

    print("Running Simulation")
    # Begin Simulation Loop
    for frame in range(frames):

        
        with torch.no_grad():
            evader_output = evader_net(evader_input)    
            pursuer_output = pursuer_net(pursuer_input)
        
        pursuer_traj = GetTrajFromBatchinput(pursuer_output,pursuer_input,pursuer_num_traj,traj_generator,solver_max_iter,device)
        pursuer_traj_ref = pursuer_traj.clone().detach()
        
        evader_traj = GetTrajFromBatchinput(evader_output,evader_input,evader_num_traj,traj_generator,solver_max_iter,device)
        evader_traj_ref = evader_traj.clone().detach()

        # Create the Bimatrix Game for the Pursuer
        pursuer_BMG_matrix = torch.zeros((pursuer_num_traj,evader_num_traj))
        for i in range(pursuer_num_traj):
            for j in range(evader_num_traj):
                pursuer_BMG_matrix[i][j] = MSE_loss(pursuer_traj[i],evader_traj_ref[j])
                
        # Create the Bimatrix Game for the Evader
        evader_BMG_matrix = torch.zeros((evader_num_traj,pursuer_num_traj))
        for i in range(evader_num_traj):
            for j in range(pursuer_num_traj):
                evader_BMG_matrix[i][j] = MSE_loss(evader_traj[i],pursuer_traj_ref[j])
                
        # Solve the Bimatrix Game
        pursuer_BMG_matrix_np = pursuer_BMG_matrix.clone().detach().numpy()
        evader_BMG_matrix_np = evader_BMG_matrix.clone().detach().numpy()
        game = nash.Game(pursuer_BMG_matrix_np, evader_BMG_matrix_np)
        equilibria = game.lemke_howson_enumeration()

        sorted_equilibria = sorted(equilibria, key=lambda x: sum(x[0] * pursuer_BMG_matrix_np @ x[1]))
        pursuer_sol = torch.tensor(sorted_equilibria[0][0], dtype=torch.float)
        evader_sol = torch.tensor(sorted_equilibria[0][1],dtype=torch.float)    
        
        # Calculate the trajectory
        pursuer_final_traj = torch.mm(pursuer_sol.view(1,-1).to(device),pursuer_traj)
        evader_final_traj = torch.mm(evader_sol.view(1,-1).to(device),evader_traj)
        pursuer_final_traj = pursuer_final_traj.squeeze()
        evader_final_traj = evader_final_traj.squeeze()
        
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

        # Check if the evader is cornered
        if (abs(evader_final_traj[0]) > (xy_limit-0.2)) and   (abs(evader_final_traj[1])> (xy_limit-0.2)):
            pursuer_states = pursuer_states[:frame+1]
            evader_states = evader_states[:frame+1]
            pursuer_trajectories = pursuer_trajectories[:frame+1]
            evader_trajectories = evader_trajectories[:frame+1]
            print("Evader Cornered")
            break
        
        print("Frame: ", frame+1 , '/', frames)
        
    print("Evader States Shape: ", evader_states.shape)
    print("Pursuer States Shape: ", pursuer_states.shape)
    print("Evader Trajectories Shape: ", evader_trajectories.shape)
    print("Pursuer Trajectories Shape: ", pursuer_trajectories.shape)

    print("----------------------------------------------")
    print("Running Animation")
    print("----------------------------------------------")

    animate(fps, name, pursuer_states, evader_states, pursuer_trajectories, evader_trajectories)
    print("Animation Complete")

if __name__ == "__main__":
    main()

