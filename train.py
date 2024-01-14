""" 

Trajectory Game Generator
Author: Avvienash Jaganathan
Date: 11/1/2024

Description : 

This file contains the code for the trajectory game generator. 

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

plt.rcParams['animation.ffmpeg_path'] = 'C:\\Users\\Avvienash\\Documents\\ffmpeg-6.1-essentials_build\\ffmpeg-6.1-essentials_build\\bin\\ffmpeg.exe'
#endregion



""" Defining the Global Variables """
#region Global Variable
# dynamics parameters
dim_x = 4 # x, y, vx, vy
dim_u = 2 # ax, ay
dt = 0.2 # time step

# Limits
xy_limit = 5
acc_limit = 0.5

# Trajectory Generator Parameters
n_steps = 10 # number of steps in the trajectory
pursuer_num_traj = 2 # number of trajectories to be generated for the pursuer
evader_num_traj = 2 # number of trajectories to be generated for the evader

# NN parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # device
input_layer_num = dim_x * 2 # pursuit and evader state
hidden_layer_num = 100 # number of neurons in the hidden layer
output_layer_num = (dim_x + dim_u) * n_steps # number of neurons in the output layer (x, y, vx, vy, ax, ay) * n_steps

# define Learning parameters
P_LR = 1e-2# learning rate for the pursuer
E_LR = 1e-2 # learning rate for the evader

# Initial State
pursuer_init_state = np.array([3,4,0,0])
evader_init_state = np.array([4,4,0,0])

# Interation Parameters
solver_max_iter = 200000 # maximum number of iterations for the solver used in the trajectory generator function
reset_bool = True # If True then the initial state is reset
reset_size = 10 # Number of iterations per episode
num_episodes = 500
total_iteration = num_episodes * reset_size# total number of iterations

# Weights for the objective function
W_state = 1000
W_control = 1

# Train Boolean
train_pursuer = True
train_evader = True
reset_evader = True

# Save Version
save_version = "3"
use_version = False
#endregion


""" Main Function """""

def main():
    # Description: This function is the main function.
    print("---------------------------------------------------------------")
    print("Running Training")    
    print("Current device:", device)  # print the current device (GPU or CPU)
    print("---------------------------------------------------------------")
    
    # Create the neural networks
    pursuer_net = PlayerTrajectoryGenerator(pursuer_num_traj,dim_x,n_steps,xy_limit,acc_limit,input_layer_num,hidden_layer_num,output_layer_num,device).to(device) # create the pursuer network 
    evader_net = PlayerTrajectoryGenerator(evader_num_traj,dim_x,n_steps,xy_limit,acc_limit,input_layer_num,hidden_layer_num,output_layer_num,device).to(device)  # create the evader network
    if use_version != False:
        pursuer_net.load_state_dict(torch.load(f'weights/pursuer_weights_v{use_version}.pth'))
        evader_net.load_state_dict(torch.load(f'weights/evader_weights_v{use_version}.pth'))
        print("Successfully loaded the trained networks")
    
    # Create the optimizer
    pursuer_optimizer = optim.SGD(pursuer_net.parameters(), lr=P_LR) # create the pursuer optimizer
    evader_optimizer = optim.SGD(evader_net.parameters(), lr=E_LR) # create the evader optimizer
    MSE_loss = nn.MSELoss()
    
    # Initial Input
    pursuer_input = torch.tensor(np.concatenate((pursuer_init_state, evader_init_state)),dtype =torch.float) # initial input for the pursuer
    evader_input  = torch.tensor(np.concatenate((evader_init_state, pursuer_init_state)),dtype =torch.float) # initial input for the evader
        
    # create Trajectory Generator
    traj_generator = construct_mpc_problem(dim_x,dim_u,n_steps,xy_limit,acc_limit,dt,W_state,W_control,output_layer_num)
    
    # get evader output
    evader_output = evader_net(evader_input)  # (num_traj, output_layer_num * n_steps)
    
    # get evader trajectory reference
    evader_traj_ref = GetTrajFromBatchinput(evader_output,evader_input,evader_num_traj,traj_generator,solver_max_iter,device) # (num_traj, dim_x * n_steps)
    
    # Create Array to store the error
    pursuer_errors = []
    evader_errors  = []
    
    
    # Begin Training Loop
    print("---------------------------------------------------------------")
    print("Begin Training Loop")
    print("---------------------------------------------------------------")
    for i_episode in range(total_iteration):
        
        # Training the pursuer
        pursuer_output = pursuer_net(pursuer_input)
        pursuer_traj = GetTrajFromBatchinput(pursuer_output,pursuer_input,pursuer_num_traj,traj_generator,solver_max_iter,device)
        pursuer_traj_ref = pursuer_traj.clone().detach()
        
        # Training the evader
        evader_output = evader_net(evader_input)
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
                evader_BMG_matrix[i][j] = -1 * MSE_loss(evader_traj[i],pursuer_traj_ref[j])
                
        # Solve the Bimatrix Game
        pursuer_BMG_matrix_np = pursuer_BMG_matrix.clone().detach().numpy()
        evader_BMG_matrix_np = evader_BMG_matrix.clone().detach().numpy()
        game = nash.Game(pursuer_BMG_matrix_np, evader_BMG_matrix_np)
        equilibria = game.lemke_howson_enumeration()


        
        def equilibria_key(item):
            if (item[0].shape[0] != pursuer_num_traj) or (item[1].shape[0] != evader_num_traj):
                print("Error: Equilibria is not the correct shape")
                return 0
            return sum(item[0] * (pursuer_BMG_matrix_np @ item[1]))

        sorted_equilibria = sorted(equilibria, key=equilibria_key)

        pursuer_sol = torch.tensor(sorted_equilibria[0][0], dtype=torch.float)
        evader_sol = torch.tensor(sorted_equilibria[0][1],dtype=torch.float)
        
        # Calculate the error
        pursuer_error = torch.mm(torch.mm(pursuer_sol.view(1,-1),pursuer_BMG_matrix),evader_sol.view(-1,1)) 
        evader_error = torch.mm(torch.mm(evader_sol.view(1,-1),evader_BMG_matrix),pursuer_sol.view(-1,1))
        
        # Store the error
        pursuer_errors.append(pursuer_error.item())
        evader_errors.append(evader_error.item())
        
        # Backpropagation
        pursuer_net.zero_grad()
        pursuer_error.backward()
        if train_pursuer:
            pursuer_optimizer.step()

        evader_net.zero_grad()
        evader_error.backward()
        if train_evader:
            evader_optimizer.step()
            
        
        # Calculate the trajectory
        pursuer_final_traj = torch.mm(pursuer_sol.view(1,-1).to(device),pursuer_traj)
        evader_final_traj = torch.mm(evader_sol.view(1,-1).to(device),evader_traj)
        pursuer_final_traj = pursuer_final_traj.squeeze()
        evader_final_traj = evader_final_traj.squeeze()
        
        
        if (i_episode % reset_size == 0) and reset_bool:
            pursuer_input = torch.tensor([*pursuer_final_traj.clone().detach()[:4],*evader_final_traj.clone().detach()[:4]], dtype=torch.float)
            evader_input = torch.tensor([*evader_final_traj.clone().detach()[:4],*pursuer_final_traj.clone().detach()[:4]], dtype=torch.float)

            
        
        # Check if Evader is caught
        if reset_evader:
            
            if (abs(evader_final_traj[0] - pursuer_final_traj[0]) <= 0.2) and  (abs(evader_final_traj[1] - pursuer_final_traj[1]) <= 0.2) :
                
                print("Evader Caught")
                evader_input = torch.tensor([random.randrange( -1* round(xy_limit*0.8) , round(xy_limit*0.8) ),
                                            random.randrange( -1 * round(xy_limit*0.8) , round(xy_limit*0.8) ),
                                            0,
                                            0,
                                            pursuer_final_traj[0],
                                            pursuer_final_traj[1],
                                            pursuer_final_traj[2],
                                            pursuer_final_traj[3]],
                                            dtype = torch.float)

            # Check if the evader is cornered
            if (abs(evader_final_traj[0]) > (xy_limit-0.2)) and   (abs(evader_final_traj[1])> (xy_limit-0.2)):
                
                print("Evader Cornered")
                evader_input = torch.tensor([random.randrange( -1* round(xy_limit*0.8) , round(xy_limit*0.8) ),
                                            random.randrange( -1 * round(xy_limit*0.8) , round(xy_limit*0.8) ),
                                            0,
                                            0,
                                            pursuer_final_traj[0],
                                            pursuer_final_traj[1],
                                            pursuer_final_traj[2],
                                            pursuer_final_traj[3]],
                                            dtype = torch.float)
        
        
        # Print the results
        print("Episode: ", i_episode+1 , '/', total_iteration , "Pursuer Error: ", pursuer_error.item(), "Evader Error: ", evader_error.item())
        
    print("Training Complete")
    
    print("---------------------------------------------------------------")
    print("plotting the results")
    print("---------------------------------------------------------------")
    
    # Plot the loss
    plt.figure(1)
    plt.plot(pursuer_errors, label='Pursuer Errors')
    plt.plot(evader_errors, label='Evader Errors')
    plt.legend()
    plt.xlabel('episode')  # Replace 'X-axis label' with the appropriate label for your data
    plt.ylabel('Error')  # Replace 'Y-axis label' with the appropriate label for your data
    plt.title('Pursuer and Evader Errors over Time')  # Replace 'Title' with the appropriate title for your plot
    plt.show()

    print("---------------------------------------------------------------")
    print("Saving the weights")
    print("---------------------------------------------------------------")
    # Save the model weights
    if save_version != False:
        torch.save(pursuer_net.state_dict(), f'weights/pursuer_weights_v{save_version}.pth')
        torch.save(evader_net.state_dict(), f'weights/evader_weights_v{save_version}.pth')
        
if __name__ == "__main__":
    main()

