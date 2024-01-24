""" 

Trajectory Game Generator
Author: Avvienash Jaganathan
Date: 11/1/2024

Description : 

This file contains the code for the trajectory game generator. 

"""

import math
import os
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
import sys 
import logging

from cvxpylayers.torch import CvxpyLayer
from cvxpy.problems.objective import Maximize, Minimize
from matplotlib.animation import FFMpegWriter
from IPython.display import Video
from datetime import datetime

# Function to get the latest version in a folder
def get_latest_version(folder,name):
    versions = [int(file.split('_v')[1].split('.')[0]) for file in os.listdir(folder) if file.startswith(name)]
    return max(versions) if versions else 0

class Genarator_NN(nn.Module):
    
    # Description: This class is a neural network model implemented using PyTorch's nn.Module for generating player trajectories.
    def __init__(self,
                 num_traj,
                 state_dim,
                 input_dim,
                 n_steps,
                 limits,
                 hidden_layer_num,
                 device):
        
        super(Genarator_NN, self).__init__() # calling the parent class constructor
        
        logging.info("Initializing Player Trajectory Generator Pytorch Model...")

        # Define the parameters
        self.num_traj = num_traj
        self.device = device
        self.n_steps = n_steps
        self.state_dim = state_dim
        self.input_dim = input_dim
       
        # Define the Mag vector for scaling
        self.mag_vector = torch.tensor(limits).repeat(1,self.num_traj*self.n_steps).to(self.device)
        
        logging.info("Mag Vector Shape: %s", self.mag_vector.shape)
        
        # Define the model
        self.model = nn.Sequential(
            nn.Linear(state_dim*2, hidden_layer_num), # input layer
            nn.Tanh(),
            nn.Linear(hidden_layer_num, hidden_layer_num), # hidden layer
            nn.Tanh(),
            nn.Linear(hidden_layer_num, hidden_layer_num), # hidden layer
            nn.Tanh(),
            nn.Linear(hidden_layer_num, hidden_layer_num), # hidden layer   
            nn.Tanh(),
            nn.Linear(hidden_layer_num, (self.state_dim + self.input_dim) * self.n_steps * self.num_traj), # output layer
            nn.Tanh()
        )
        
    def forward(self, x):

        x = x.to(self.device) # send the input to the device
        x = self.model(x).mul(self.mag_vector) # forward pass
        x = x.view(self.num_traj,self.n_steps,-1) # reshape the output
        return x

class PlayerTrajectoryGenerator:
    """ 
    Description: This class is the main class for the trajectory game generator. 
    """
    def __init__(self,
                 num_traj,
                 state_dim,
                 input_dim,
                 n_steps,
                 dt,
                 limits,
                 hidden_layer_num,
                 solver_max_iter,
                 device,
                 verbose = False,
                 solve_method = 'ECOS'):
        
        # Define the parameters
        self.num_traj = num_traj
        self.device = device
        self.n_steps = n_steps
        self.dt = dt
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.limits = limits
        self.hidden_layer_num = hidden_layer_num
        self.solver_args =  {
                            "max_iters": solver_max_iter,
                            "verbose": verbose, 
                            'solve_method':solve_method,
                            }
        self.MSE_loss = nn.MSELoss()
        
        # Define the models
        self.pursuer_model = Genarator_NN(num_traj,
                                  state_dim,
                                  input_dim,
                                  n_steps,
                                  limits,
                                  hidden_layer_num,
                                  device).to(self.device)
        
        self.evader_model = Genarator_NN(num_traj,
                                  state_dim,
                                  input_dim,
                                  n_steps,
                                  limits,
                                  hidden_layer_num,
                                  device).to(self.device)
        
    def contruct_optimization_problem(self):
        
        # dynamics 
        A = np.array([[1, 0, self.dt, 0],
                        [0, 1, 0, self.dt],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

        B = np.array([[self.dt * self.dt * 0.5, 0],
                        [0, self.dt * self.dt * 0.5],
                        [self.dt, 0],
                        [0, self.dt]])
        
        # Define Parameters
        current_state_param = cp.Parameter(self.state_dim)
        current_traj_param = cp.Parameter((self.n_steps, self.state_dim + self.input_dim))
        
        # Define Variables
        init_state_var = cp.Variable(self.state_dim)
        state_var = cp.Variable((self.n_steps,self.state_dim))
        input_var = cp.Variable((self.n_steps,self.input_dim))
        
        # Define Objectives
        objective = cp.norm2(100 * (init_state_var - current_state_param).reshape(-1))
        objective += cp.norm2(10 *(state_var - current_traj_param[:,:self.state_dim]).reshape(-1) )
        objective += cp.norm2(1 * (input_var - current_traj_param[:,self.state_dim:]).reshape(-1) )  
        
        # define constraints
        constraints = [state_var[0,:] == A @ init_state_var + B @ input_var[0,:]]
        
        constraints += [state_var[0] <= self.limits[0]]
        constraints += [state_var[1] <= self.limits[1]]
        constraints += [state_var[2] <= self.limits[2]]
        constraints += [state_var[3] <= self.limits[3]]
        
        constraints += [state_var[0] >= -1*self.limits[0]]
        constraints += [state_var[1] >= -1*self.limits[1]]
        constraints += [state_var[2] >= -1*self.limits[2]]
        constraints += [state_var[3] >= -1*self.limits[3]]
        
        for i in range(self.n_steps):
            
            if i > 0:
                constraints += [state_var[i,:] == A @ state_var[i-1,:] + B @ input_var[i,:]]
        
            constraints += [state_var[i,0] <= self.limits[0]]
            constraints += [state_var[i,1] <= self.limits[1]]
            constraints += [state_var[i,2] <= self.limits[2]]
            constraints += [state_var[i,3] <= self.limits[3]]
            
            constraints += [state_var[i,0] >= -1*self.limits[0]]
            constraints += [state_var[i,1] >= -1*self.limits[1]]
            constraints += [state_var[i,2] >= -1*self.limits[2]]
            constraints += [state_var[i,3] >= -1*self.limits[3]]
            
            constraints += [input_var[i,0] <= self.limits[4]]
            constraints += [input_var[i,1] <= self.limits[5]]
            
            constraints += [input_var[i,0] >= -1*self.limits[4]]
            constraints += [input_var[i,1] >= -1*self.limits[5]]

            
        problem = cp.Problem(cp.Minimize(objective), constraints)
        assert problem.is_dcp()
        assert problem.is_dpp()
        
        return CvxpyLayer(problem, variables=[init_state_var,state_var,input_var], parameters=[current_state_param,current_traj_param])
            
    def convex_optimization_pass(self, traj, current_state, optimization_layer):
        
        """
        Description: This function solves the optimization dynamics using CVXPY.
        """
        optimized_init_state = []
        optimzed_states = []
        optimized_inputs = []
        
        for i in range(self.num_traj):
            
            init_state, states, inputs = optimization_layer(current_state.to(self.device),
                                                            traj[i,:,:].to(self.device), 
                                                            solver_args= self.solver_args)
            optimized_init_state.append(init_state)
            optimzed_states.append(states)
            optimized_inputs.append(inputs)
        
        optimized_init_state = torch.stack(optimized_init_state,dim = 0)
        optimzed_states = torch.stack(optimzed_states,dim = 0)
        optimized_inputs = torch.stack(optimized_inputs,dim = 0)
        
        return optimized_init_state, optimzed_states, optimized_inputs
    
    def generate_trajectories(self, pursuer_current_state, evader_current_state, optimization_layer):
        
        # Initial Input
        pursuer_input = torch.cat((pursuer_current_state, evader_current_state), dim=0).float()
        evader_input = torch.cat((evader_current_state, pursuer_current_state), dim=0).float()
        
        logging.debug("Pursuer Input: %s", pursuer_input)
        logging.debug("Evader Input: %s", evader_input)
        logging.debug("Puruser Input Shape: %s", pursuer_input.shape)
        logging.debug("Evader Input Shape: %s", evader_input.shape)
        
        pursuer_output = self.pursuer_model(pursuer_input.to(self.device))
        evader_output = self.evader_model(evader_input.to(self.device))
        
        logging.debug("Pursuer Output: %s", pursuer_output)
        logging.debug("Evader Output: %s", evader_output)
        logging.debug("Pursuer Output Shape: %s", pursuer_output.shape)
        logging.debug("Evader Output Shape: %s", evader_output.shape)
        
        pursuer_output_optimized_init_state, pursuer_output_optimized_states, pursuer_output_optimized_inputs = self.convex_optimization_pass(traj = pursuer_output, 
                                                                                                                                              current_state = pursuer_current_state,
                                                                                                                                              optimization_layer = optimization_layer)
        
        evader_output_optimized_init_state, evader_output_optimized_states, evader_output_optimized_inputs = self.convex_optimization_pass(traj = evader_output, 
                                                                                                                                              current_state = evader_current_state,
                                                                                                                                              optimization_layer = optimization_layer)

        logging.debug("Pursuer Output Optimized Init State: %s", pursuer_output_optimized_init_state)
        logging.debug("Pusuer Output Optimized States: %s", pursuer_output_optimized_states)
        logging.debug("Pursuer Output Optimized Inputs: %s", pursuer_output_optimized_inputs)
        
        logging.debug("Pursuer Output Optimized Init State Shape: %s", pursuer_output_optimized_init_state.shape)
        logging.debug("Pusuer Output Optimized States Shape: %s", pursuer_output_optimized_states.shape)
        logging.debug("Pursuer Output Optimized Inputs Shape: %s", pursuer_output_optimized_inputs.shape)
        
        
        pursuer_output_optimized_states_ref = pursuer_output_optimized_states.clone().detach()
        evader_output_optimized_states_ref = evader_output_optimized_states.clone().detach()
        
        pursuer_BMG_matrix = torch.zeros(self.num_traj,self.num_traj)
        for i in range(self.num_traj):
            for j in range(self.num_traj):
                pursuer_BMG_matrix[i,j] = self.MSE_loss(pursuer_output_optimized_states[i,:,:],evader_output_optimized_states_ref[j,:,:])
                
        evader_BMG_matrix = torch.zeros(self.num_traj,self.num_traj)
        for i in range(self.num_traj):
            for j in range(self.num_traj):
                evader_BMG_matrix[i,j] = -1 * self.MSE_loss(evader_output_optimized_states[i,:,:],pursuer_output_optimized_states_ref[j,:,:])
                
        pursuer_BMG_matrix_np = pursuer_BMG_matrix.clone().detach().numpy()
        evader_BMG_matrix_np = evader_BMG_matrix.clone().detach().numpy()
        
        logging.debug("Pursuer BMG Matrix: %s", pursuer_BMG_matrix)
        logging.debug("Pursuer BMG Matrix Shape: %s", pursuer_BMG_matrix.shape)
        logging.debug("Evader BMG Matrix: %s", evader_BMG_matrix)
        logging.debug("Evader BMG Matrix Shape: %s", evader_BMG_matrix.shape)
        
        game = nash.Game(pursuer_BMG_matrix_np, evader_BMG_matrix_np)
        equilibria = game.support_enumeration()
        
        sorted_equilibria = sorted(equilibria, key=lambda item: sum(item[0] * (pursuer_BMG_matrix_np @ item[1])))
        
        logging.debug("Equilibria: %s", sorted_equilibria)
        logging.debug("Equilibria Shape: %s", len(sorted_equilibria))
        
        pursuer_sol = torch.tensor(sorted_equilibria[0][0], dtype=torch.float)
        evader_sol = torch.tensor(sorted_equilibria[0][1],dtype=torch.float)
        
        logging.debug("Pursuer Solution: %s", pursuer_sol)
        logging.debug("Evader Solution: %s", evader_sol)
        logging.debug("Pursuer Solution Shape: %s", pursuer_sol.shape)
        logging.debug("Evader Solution Shape: %s", evader_sol.shape)
        
        # Calculate the Error:
        pursuer_error = torch.mm(torch.mm(pursuer_sol.view(1,-1),pursuer_BMG_matrix),evader_sol.view(-1,1)) 
        evader_error = torch.mm(torch.mm(evader_sol.view(1,-1),evader_BMG_matrix),pursuer_sol.view(-1,1))
        
        logging.debug("Pursuer Error: %s", pursuer_error)
        logging.debug("Evader Error: %s", evader_error)
        logging.debug("Pursuer Error Shape: %s", pursuer_error.shape)
        logging.debug("Evader Error Shape: %s", evader_error.shape)
        
        # Calculate the Trajectories
        # pursuer_final_states = torch.mm(pursuer_sol.view(1,-1).to(self.device),pursuer_output_optimized_states)
        # evader_final_states = torch.mm(evader_sol.view(1,-1).to(self.device),evader_output_optimized_states)
        pursuer_sol_index = torch.argmax(pursuer_sol).item()
        evader_sol_index = torch.argmax(evader_sol).item()
        
        pursuer_final_states = pursuer_output_optimized_states[pursuer_sol_index,:,:]
        evader_final_states = evader_output_optimized_states[evader_sol_index,:,:]
        
        logging.debug("Pursuer Final States: %s", pursuer_final_states)
        logging.debug("Evader Final States: %s", evader_final_states)
        logging.debug("Pursuer Final States Shape: %s", pursuer_final_states.shape)
        logging.debug("Evader Final States Shape: %s", evader_final_states.shape)
        
        
        return pursuer_final_states, evader_final_states, pursuer_error, evader_error
    
    def train(self,P_LR, E_LR, num_epochs, save_model = True, reset_n_step = 1000):
        
        logging.info("Training Player Trajectory Generator Pytorch Model...")
        
        pursuer_optimizer = optim.SGD(self.pursuer_model.parameters(), lr=P_LR) # create the pursuer optimizer
        evader_optimizer = optim.SGD(self.evader_model.parameters(), lr=E_LR)
        
        optimization_layer = self.contruct_optimization_problem()
        
        logging.debug("Optimization Layer: %s", optimization_layer)
        
        # Initialize the states_for sim
        pursuer_error_sim = np.zeros((num_epochs,1))
        evader_error_sim = np.zeros((num_epochs,1))
        pursuer_states_sim = np.zeros((num_epochs, self.state_dim))
        evader_states_sim = np.zeros((num_epochs, self.state_dim))
        pursuer_trajectories_sim = np.zeros((num_epochs, self.n_steps, self.state_dim))
        evader_trajectories_sim = np.zeros((num_epochs, self.n_steps, self.state_dim))
        
        
        pursuer_init_state = torch.randn(self.state_dim).mul(torch.tensor(self.limits[:self.state_dim])).to(self.device)
        evader_init_state = torch.randn(self.state_dim).mul(torch.tensor(self.limits[:self.state_dim])).to(self.device)
        
        logging.info("Pursuer Initial State: %s", pursuer_init_state)
        logging.info("Evader Initial State: %s", evader_init_state)
        
        self.pursuer_model.train()
        self.evader_model.train()
        
        reset_counter = 0
        
        for epoch in range(num_epochs):
            
            pursuer_final_states, evader_final_states, pursuer_error, evader_error = self.generate_trajectories(pursuer_init_state, evader_init_state, optimization_layer)
            
            pursuer_optimizer.zero_grad() # zero the gradients
            pursuer_error.backward()
            pursuer_optimizer.step()
            
            evader_optimizer.zero_grad() # zero the gradients
            evader_error.backward()
            evader_optimizer.step()
            
            # Update the states
            pursuer_error_sim[epoch] = pursuer_error.clone().detach().cpu().numpy()
            evader_error_sim[epoch] = evader_error.clone().detach().cpu().numpy()
            pursuer_states_sim[epoch,:] = pursuer_init_state.clone().detach().cpu().numpy()
            evader_states_sim[epoch,:] = evader_init_state.clone().detach().cpu().numpy()
            pursuer_trajectories_sim[epoch,:,:] = pursuer_final_states.clone().detach().cpu().numpy()
            evader_trajectories_sim[epoch,:,:] = evader_final_states.clone().detach().cpu().numpy()
            
            
            pursuer_init_state = pursuer_final_states[0,:].clone().detach()
            evader_init_state = evader_final_states[0,:].clone().detach()
            
            
            # check if evader caught by pursuer
            if torch.norm(pursuer_init_state[:2] - evader_init_state[:2]) < 0.4:
                
                logging.info("Evader Caught by Pursuer")
                pursuer_init_state = torch.randn(self.state_dim).mul(torch.tensor(self.limits[:self.state_dim])).to(self.device)
                evader_init_state = torch.randn(self.state_dim).mul(torch.tensor(self.limits[:self.state_dim])).to(self.device)
                reset_counter = 0

            # reset the states after 1000 epochs
            reset_counter += 1
            if reset_counter == reset_n_step:
                logging.info("Resetting the states afteer 1000 steps of evader not being caught by pursuer")
                pursuer_init_state = torch.randn(self.state_dim).mul(torch.tensor(self.limits[:self.state_dim])).to(self.device)
                evader_init_state = torch.randn(self.state_dim).mul(torch.tensor(self.limits[:self.state_dim])).to(self.device)
                reset_counter = 0
            
            
            logging.info("Epoch: %s, Pursuer Error: %s, Evader Error: %s, Distance between pursuer and evader: %s",
                        epoch, pursuer_error_sim[epoch], evader_error_sim[epoch],
                        torch.norm(pursuer_init_state[:2] - evader_init_state[:2]).item())

        # Save the models
        if save_model:
            
            version = get_latest_version("models","pursuer_model_v") + 1
            
            pursuer_model_name = "models/pursuer_model_v" + str(version) + ".pth"
            evader_model_name = "models/evader_model_v" + str(version) + ".pth"
            
            torch.save(self.pursuer_model.state_dict(), pursuer_model_name) 
            torch.save(self.evader_model.state_dict(), evader_model_name)
            
            # save the params
            params = {
                'num_traj': self.num_traj,
                'state_dim': self.state_dim,
                'input_dim': self.input_dim,
                'n_steps': self.n_steps,
                'dt': self.dt,
                'limits': self.limits,
                'hidden_layer_num': self.hidden_layer_num,
                'solver_max_iter': self.solver_args['max_iters'],
                'device': self.device,
                'verbose': False,
                'solve_method': self.solver_args['solve_method']
            }
            
            with open(f'models/model_params_v{version}.txt', 'w') as f:
                for key, value in params.items():
                    f.write(f'{key}: {value}\n')
        
        return pursuer_error_sim, evader_error_sim, pursuer_states_sim, evader_states_sim, pursuer_trajectories_sim, evader_trajectories_sim
    
    def test(self, pursuer_init_state, evader_init_state, num_epochs):
        
        logging.info("Testing Player Trajectory Generator Pytorch Model...")
        
        optimization_layer = self.contruct_optimization_problem()
        
        logging.info("Pursuer Initial State: %s", pursuer_init_state)
        logging.info("Evader Initial State: %s", evader_init_state)
        
        pursuer_states_sim = np.zeros((num_epochs, self.state_dim))
        evader_states_sim = np.zeros((num_epochs, self.state_dim))
        pursuer_trajectories_sim = np.zeros((num_epochs, self.n_steps, self.state_dim))
        evader_trajectories_sim = np.zeros((num_epochs, self.n_steps, self.state_dim))
        
        self.pursuer_model.eval()
        self.evader_model.eval()
        
        for epoch in range(num_epochs):
            
            with torch.no_grad():
                pursuer_final_states, evader_final_states, pursuer_error, evader_error = self.generate_trajectories(pursuer_init_state, evader_init_state, optimization_layer)
            
            # Update the states
            pursuer_states_sim[epoch,:] = pursuer_init_state.clone().detach().cpu().numpy()
            evader_states_sim[epoch,:] = evader_init_state.clone().detach().cpu().numpy()
            pursuer_trajectories_sim[epoch,:,:] = pursuer_final_states.clone().detach().cpu().numpy()
            evader_trajectories_sim[epoch,:,:] = evader_final_states.clone().detach().cpu().numpy()
            
            # check if evader caught by pursuer
            pursuer_init_state = pursuer_final_states[0,:].clone().detach()
            evader_init_state = evader_final_states[0,:].clone().detach()
            
            if torch.norm(pursuer_init_state[:2] - evader_init_state[:2]) < 0.4:
                
                logging.info("Evader Caught by Pursuer")
                # remove remaining states
                pursuer_states_sim = pursuer_states_sim[:epoch,:]
                evader_states_sim = evader_states_sim[:epoch,:]
                pursuer_trajectories_sim = pursuer_trajectories_sim[:epoch,:,:]
                evader_trajectories_sim = evader_trajectories_sim[:epoch,:,:]
                break

            
            logging.info("Epoch: %s, Pursuer Error: %s, Evader Error: %s, Distance between pursuer and evader: %s",
                        epoch, pursuer_error, evader_error,
                        torch.norm(pursuer_init_state[:2] - evader_init_state[:2]).item())
        
        return pursuer_states_sim, evader_states_sim, pursuer_trajectories_sim, evader_trajectories_sim
    
    def load_model(self, version):
        
        pursuer_model_name = "models/pursuer_model_v" + str(version) + ".pth"
        evader_model_name = "models/evader_model_v" + str(version) + ".pth"
        
        self.pursuer_model.load_state_dict(torch.load(pursuer_model_name))
        self.evader_model.load_state_dict(torch.load(evader_model_name))
        
        return
    
    def plot_losses(self, pursuer_losses, evader_losses):
        """
        Description: Plot the training losses for both pursuer and evader.

        Parameters:
            - pursuer_losses: NumPy array containing pursuer losses for each epoch.
            - evader_losses: NumPy array containing evader losses for each epoch.
        """
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(pursuer_losses) + 1)

        plt.plot(epochs, pursuer_losses, label='Pursuer Loss')
        plt.plot(epochs, evader_losses, label='Evader Loss')

        plt.title('Training Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plot_name = "plots/losses_plot_v" + str(get_latest_version('plots',"losses_plot_v") + 1) + ".png"
        plt.savefig(plot_name)
        
    def animate(self,pursuer_states, evader_states, pursuer_trajectories, evader_trajectories, name):
        
        plt.rcParams['animation.ffmpeg_path'] = 'C:\\Users\\Avvienash\\Documents\\ffmpeg-6.1-essentials_build\\ffmpeg-6.1-essentials_build\\bin\\ffmpeg.exe'

        fps = 10

        fig = plt.figure(figsize=(5, 5))
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

        
        metadata = dict(title='demo', artist='avvienash')
        writer = FFMpegWriter(fps=fps, metadata=metadata)      

        with writer.saving(fig, name, 100):
            
            for i in range(evader_states.shape[0]):

                pursuer_traj_plot_x = pursuer_trajectories[i,:,0]
                pursuer_traj_plot_y = pursuer_trajectories[i,:,1]
                pursuer_traj_plot.set_data(pursuer_traj_plot_x, pursuer_traj_plot_y)
                
                evader_traj_plot_x = evader_trajectories[i,:,0]
                evader_traj_plot_y = evader_trajectories[i,:,1]
                evader_traj_plot.set_data(evader_traj_plot_x, evader_traj_plot_y)
                
                pursuer_pos_plot_x = pursuer_states[i, 0]
                pursuer_pos_plot_y = pursuer_states[i, 1]
                pursuer_pos_plot.set_data([pursuer_pos_plot_x], [pursuer_pos_plot_y])
                
                evader_pos_plot_x = evader_states[i, 0]
                evader_pos_plot_y = evader_states[i, 1]
                evader_pos_plot.set_data([evader_pos_plot_x], [evader_pos_plot_y])
                
                logging.info("Animation Frame: %s", i)
                
                writer.grab_frame()
    
        return
        
def main():
    
    logs_version = get_latest_version('logs','logs')
    logs_filename = 'logs/logs_v' + str(logs_version + 1) + '.log'
    logging.basicConfig(level=logging.INFO, 
                        format='[%(levelname)s] %(message)s',
                        handlers=[ logging.StreamHandler(),logging.FileHandler(logs_filename) ])

    
    
    generator = PlayerTrajectoryGenerator(num_traj = 5,
                                          state_dim = 4,
                                          input_dim = 2,
                                          n_steps = 5,
                                          dt = 0.2,
                                          limits = [5,5,5,5,0.5,0.5],
                                          hidden_layer_num = 256,
                                          solver_max_iter = 10000,
                                          device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    pursuer_error_sim,evader_error_sim,\
    pursuer_states_sim, evader_states_sim,\
    pursuer_trajectories_sim, evader_trajectories_sim = generator.train(P_LR = 0.0001,
                                                                        E_LR = 0.0001,
                                                                        num_epochs = 200000,
                                                                        save_model = True,
                                                                        reset_n_step = 300)

    generator.plot_losses(pursuer_error_sim, evader_error_sim)

    name = "training_animations/animation_v" + str(get_latest_version('training_animations', "animation_v") + 1) + ".mp4"
    generator.animate(pursuer_states_sim, evader_states_sim, pursuer_trajectories_sim, evader_trajectories_sim, name)
    
    # Test the model
    generator.load_model(get_latest_version('models','pursuer_model_v'))
    
    pursuer_init_state = torch.tensor([0,0,0,0]).float()
    evader_init_state = torch.tensor([4,4,0,0]).float()
    
    pursuer_states_sim, evader_states_sim, pursuer_trajectories_sim, evader_trajectories_sim = generator.test(pursuer_init_state,
                                                                                                              evader_init_state,
                                                                                                              num_epochs = 200)
    
    name = "testing_animations/animation_v" + str(get_latest_version('testing_animations', "animation_v") + 1) + ".mp4"
    generator.animate(pursuer_states_sim, evader_states_sim, pursuer_trajectories_sim, evader_trajectories_sim, name)
    
    return
    
    
if __name__ == "__main__":
    main()
