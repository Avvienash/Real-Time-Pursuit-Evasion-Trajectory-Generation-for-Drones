"""
Utility functions for trajectory generation using PyTorch and CVXPY.
Author: Avvienash
Date: 12/1/2024

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
from matplotlib.gridspec import GridSpec


from cvxpylayers.torch import CvxpyLayer
from matplotlib.animation import FFMpegWriter
from IPython.display import Video

plt.rcParams['animation.ffmpeg_path'] = 'C:\\Users\\Avvienash\\Documents\\ffmpeg-6.1-essentials_build\\ffmpeg-6.1-essentials_build\\bin\\ffmpeg.exe'
#endregion

""" Defining the Classes """

class PlayerTrajectoryGenerator(nn.Module):
    
    # Description: This class is a neural network model implemented using PyTorch's nn.Module for generating player trajectories.

    def __init__(self,num_traj,dim_x,n_steps,xy_limit,acc_limit,input_layer_num,hidden_layer_num,output_layer_num,device):
        
        super(PlayerTrajectoryGenerator, self).__init__() # calling the parent class constructor
        
        self.num_traj = num_traj
        self.device = device
       
        # Define the Mag vector for scaling
        mag_vector = torch.ones(1*output_layer_num).to(self.device)
        mag_vector[0:dim_x*n_steps] = xy_limit
        mag_vector[dim_x*n_steps:] = acc_limit
        self.mag_vector = mag_vector.repeat(1,self.num_traj)
        
        
        # Define the model
        self.model = nn.Sequential(
            nn.Linear(input_layer_num, hidden_layer_num), # input layer
            nn.Tanh(),
            nn.Linear(hidden_layer_num, hidden_layer_num), # hidden layer
            nn.Tanh(),
            nn.Linear(hidden_layer_num, hidden_layer_num), # hidden layer
            nn.Tanh(),
            nn.Linear(hidden_layer_num, hidden_layer_num), # hidden layer   
            nn.Tanh(),
            nn.Linear(hidden_layer_num, output_layer_num * self.num_traj), # output layer
            nn.Tanh()
        )
        
    def forward(self, x):

        x = x.to(self.device) # send the input to the device
        x = self.model(x).mul(self.mag_vector) # forward pass
        x = x.view(self.num_traj,-1) # reshape the output

        return x
    
    
""" Defining the Helper Functions """

def construct_mpc_problem(dim_x,dim_u,n_steps,xy_limit,acc_limit,dt,W_state,W_control,output_layer_num):
    
    T = n_steps # timest
    beta = acc_limit # acc limit

    # dynamics 
    A_np = np.array([[1, 0, dt, 0],
                    [0, 1, 0, dt],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    B_np = np.array([[dt * dt * 0.5, 0],
                    [0, dt * dt * 0.5],
                    [dt, 0],
                    [0, dt]])
    
    # Define Parameters
    x0 = cp.Parameter(dim_x) # current pose.
    x = cp.Parameter(output_layer_num) # reference trajectory
    
    # Define Variables
    states = cp.Variable(dim_x*T)
    controls = cp.Variable(dim_u*T)

    # Define the objective
    objective  = cp.norm2( W_state * (states-x[0:dim_x*T])) # Penalises diffrence between the current state and the reference trajectory
    objective += cp.norm2( W_control *(controls-x[dim_x*T:(dim_x+dim_u)*T])) # Penalises diffrence between the current control and the reference trajectory

    # Define the constraints
    constraints = [states[0:dim_x] == A_np @ x0 + B_np @ controls[0:dim_u]]
    for t in range(T):
        if t > 0:
            constraints += [states[dim_x*t:dim_x*(t+1)] == A_np @ states[dim_x*(t-1):dim_x*t] + B_np @ controls[dim_u*t:dim_u*(t+1)]]

        # xy range
        constraints += [states[dim_x*t+0]<=xy_limit]
        constraints += [states[dim_x*t+0]>=-xy_limit]
        constraints += [states[dim_x*t+1]<=xy_limit]
        constraints += [states[dim_x*t+1]>=-xy_limit]
        
        # u range
        constraints += [controls[dim_u*t] <=  beta]
        constraints += [controls[dim_u*t] >= -beta]
        constraints += [controls[dim_u*t+1] <=  beta]
        constraints += [controls[dim_u*t+1] >= -beta]

    # Define the problem
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return_layer = CvxpyLayer(problem, variables=[states, controls], parameters=[x, x0])

    return return_layer

def GetTrajFromBatchinput(nn_output,nn_input,num_traj,traj_generator,solver_max_iter,device):

    result_list = [] # list to store the result
    
    # iterate through the trajectories
    for i in range(num_traj):
        (current_traj_output,control) = traj_generator(nn_output[i],nn_input[[0,1,2,3]].to(device),solver_args={"max_iters": solver_max_iter,"verbose": False})
        result_list.append(current_traj_output.view(1,-1))
        
    result = torch.cat(result_list,dim=0) # concatenate the result to a tensor
    return result

def animate(fps, name, pursuer_states, evader_states, pursuer_trajectories, evader_trajectories):
    
    fig = plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    pursuer_pos_plot, = plt.plot([], [], 'ro', markersize=8)
    pursuer_traj_plot, = plt.plot([], [], 'r-')

    evader_pos_plot, = plt.plot([], [], 'bo',markersize=8)
    evader_traj_plot, = plt.plot([], [], 'b-')
    
    # Plot a border at xy limits of -5 to 5
    border_x = [-5, 5, 5, -5, -5]
    border_y = [-5, -5, 5, 5, -5]
    plt.plot(border_x, border_y, 'k-', linewidth=1)

    plt.xlabel('x position')
    plt.ylabel('y position')
    plt.title('Top Down Simulation')
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    

    plt.subplot(2,2,2)
    pursuer_vel_plot = plt.quiver([], [], [], [], color='r', units='xy', scale=1)
    plt.xlabel('x velocity')
    plt.ylabel('y velocity')
    plt.title('Pursuer Velocity')
    plt.axis('square')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    
    plt.subplot(2,2,4)
    evader_vel_plot = plt.quiver([], [], [], [], color='b', units='xy', scale=1)
    plt.xlabel('x velocity')
    plt.ylabel('y velocity')
    plt.title('Evader Velocity')
    plt.axis('square')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    
    plt.tight_layout()
    
    
    

    metadata = dict(title='demo', artist='avvienash')
    writer = FFMpegWriter(fps=fps, metadata=metadata)      

    with writer.saving(fig, name, 100):
        
        for i in range(evader_states.shape[0]):
            
            
            
            pursuer_traj_plot_x = pursuer_trajectories[i, 0::4]
            pursuer_traj_plot_y = pursuer_trajectories[i, 1::4]
            pursuer_traj_plot.set_data(pursuer_traj_plot_x, pursuer_traj_plot_y)
            
            evader_traj_plot_x = evader_trajectories[i, 0::4]
            evader_traj_plot_y = evader_trajectories[i, 1::4]
            evader_traj_plot.set_data(evader_traj_plot_x, evader_traj_plot_y)
            
            pursuer_pos_plot_x = pursuer_states[i, 0]
            pursuer_pos_plot_y = pursuer_states[i, 1]
            pursuer_pos_plot.set_data([pursuer_pos_plot_x], [pursuer_pos_plot_y])
            
            evader_pos_plot_x = evader_states[i, 0]
            evader_pos_plot_y = evader_states[i, 1]
            evader_pos_plot.set_data([evader_pos_plot_x], [evader_pos_plot_y])
            
            pursuer_vel_plot_plot_x = pursuer_states[i, 2]
            pursuer_vel_plot_plot_y = pursuer_states[i, 3]
            pursuer_vel_plot.set_offsets([0, 0])
            pursuer_vel_plot.set_UVC(pursuer_vel_plot_plot_x, pursuer_vel_plot_plot_y)

            
            evader_vel_plot_plot_x = evader_states[i, 2]
            evader_vel_plot_plot_y = evader_states[i, 3]
            evader_vel_plot.set_offsets([0, 0])
            evader_vel_plot.set_UVC(evader_vel_plot_plot_x, evader_vel_plot_plot_y)
            
            # Calculate distance between pursuer and evader
            distance = ((evader_pos_plot_x - pursuer_pos_plot_x)**2 + (evader_pos_plot_y - pursuer_pos_plot_y)**2)**0.5

            print("Animation Frame: ", i+1 , '/', evader_states.shape[0], "Distance: ", distance)
            writer.grab_frame()