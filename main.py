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
import torch.optim.lr_scheduler as lr_scheduler
import cvxpy as cp
import nashpy as nash
import sys 
import logging


from cvxpylayers.torch import CvxpyLayer
from cvxpy.problems.objective import Maximize, Minimize
from matplotlib.animation import FFMpegWriter
from matplotlib import cm
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
        self.mag_vector = torch.tensor(limits[-self.input_dim:]).repeat(1,self.num_traj*self.n_steps).to(self.device)
        # logging.info("Mag Vector Shape: %s", self.mag_vector.shape)
        # logging.info("Mag Vector: %s", self.mag_vector)
        
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
            nn.Linear(hidden_layer_num, (self.input_dim) * self.n_steps * self.num_traj), # output layer
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
                 pursuer_limits,
                 evader_limits,
                 catch_radius,
                 hidden_layer_num,
                 solver_max_iter,
                 save_path,
                 device,
                 verbose = False,
                 solve_method = 'ECOS',
                 enviroment = 'square',
                 margin = 1,
                 bounds_type = 'hard',):
        
        # Define the parameters
        self.num_traj = num_traj
        self.device = device
        self.n_steps = n_steps
        self.dt = dt
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.pursuer_limits = pursuer_limits
        self.evader_limits = evader_limits
        self.catch_radius = catch_radius
        self.hidden_layer_num = hidden_layer_num
        self.solver_args =  {
                            "max_iters": solver_max_iter,
                            "verbose": verbose, 
                            'solve_method':solve_method,
                            }
        self.MSE_loss = nn.MSELoss()
        self.save_path = save_path
        self.enviroment = enviroment
        self.margin = margin
        self.bounds_type = bounds_type
        
        # Define the models
        self.pursuer_model = Genarator_NN(num_traj,
                                  state_dim,
                                  input_dim,
                                  n_steps,
                                  pursuer_limits,
                                  hidden_layer_num,
                                  device).to(self.device)
        
        self.evader_model = Genarator_NN(num_traj,
                                  state_dim,
                                  input_dim,
                                  n_steps,
                                  evader_limits,
                                  hidden_layer_num,
                                  device).to(self.device)
           
    def contruct_optimization_problem(self,limits, bounds_type):
        
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
        current_traj_param = cp.Parameter((self.n_steps, self.input_dim))
        
        # Define Limits for the position boundary:
        if self.enviroment == 'square':
            boundary = [0, 0]
        elif self.enviroment == 'right_triangle':
            boundary = [0, 0, 0]
        elif self.enviroment == 'pentagon':
            boundary = [0, 0, 0, 0, 0]
            m = np.zeros((5,))
            c = np.zeros((5,))
            
            for i in range(len(m)):
                r = self.pursuer_limits[0]
                m[i] = ( (np.cos((i+1)*(2*np.pi/5)) - np.cos((i)*(2*np.pi/5))) / ( np.sin((i+1)*(2*np.pi/5)) - np.sin((i)*(2*np.pi/5)) ) )
                c[i] = r*np.cos((i)*(2*np.pi/5)) - m[i] * r* np.sin((i)*(2*np.pi/5))
        
        # Define Variables
        state_var = cp.Variable((self.n_steps,self.state_dim))
        input_var = cp.Variable((self.n_steps,self.input_dim))  
        boundary_var = cp.Variable((self.n_steps, len(boundary)))
        
        # Define Objectives
        objective = cp.norm2(1 * (input_var - current_traj_param).reshape(-1) ) 
        constraints = [state_var[0,:] == A @ current_state_param + B @ input_var[0,:]] 
        
        if bounds_type == 'hard':
            pass          
        elif bounds_type == 'soft':
            objective += cp.norm2(5 * (boundary_var).reshape(-1) )
                            
        for i in range(self.n_steps):
            
            if i > 0:
                constraints += [state_var[i,:] == A @ state_var[i-1,:] + B @ input_var[i,:]]
                
            
            for j in range(len(boundary)):
                
                if bounds_type == 'soft':
                    constraints += [boundary_var[i,j] <= boundary[j] + self.margin]
                    constraints += [boundary_var[i,j] >= -1*(boundary[j] + self.margin)]
                elif bounds_type == 'hard':
                    constraints += [boundary_var[i,j] == boundary[j]]
                
            # Ensure that the state is within the boundary's hard constraints
            if self.enviroment == 'square':
                # Square
                # Soft Constraints
                constraints += [state_var[i,0] - 5 <= boundary_var[i,0]]
                constraints += [state_var[i,1] - 5 <= boundary_var[i,1]]
                
                constraints += [state_var[i,0] + 5 >= -1*boundary_var[i,0]]
                constraints += [state_var[i,1] + 5 >= -1*boundary_var[i,1]]
                
                
            elif self.enviroment == 'right_triangle':

                # RIght Triangle
                constraints += [state_var[i,0] - 5 <= boundary_var[i,0]]
                constraints += [state_var[i,1] -  5 <= boundary_var[i,1]]
                
                constraints += [state_var[i,0] + 5 >= -1*boundary_var[i,0]]
                constraints += [state_var[i,1] + 5 >= -1*boundary_var[i,1]]
                
                constraints += [state_var[i,0] - state_var[i,1] >= boundary_var[i,2]]
                
            elif self.enviroment == 'pentagon':
                
                # Pentagon
                constraints += [state_var[i,1] - state_var[i,0]*m[0] - c[0] <= boundary_var[i,0]]
                constraints += [state_var[i,1] - state_var[i,0]*m[1] - c[1] >= boundary_var[i,1]]
                constraints += [state_var[i,1] - state_var[i,0]*m[2] - c[2] >= boundary_var[i,2]]
                constraints += [state_var[i,1] - state_var[i,0]*m[3] - c[3] >= boundary_var[i,3]]
                constraints += [state_var[i,1] - state_var[i,0]*m[4] - c[4] <= boundary_var[i,4]]
                
            # Vel Constraints
            constraints += [state_var[i,2] <= limits[2]]
            constraints += [state_var[i,3] <= limits[3]]
            
            constraints += [state_var[i,2] >= -1*limits[2]]
            constraints += [state_var[i,3] >= -1*limits[3]]
            
            # Acc Constraints
            constraints += [input_var[i,0] <= limits[4]]
            constraints += [input_var[i,1] <= limits[5]]
            
            constraints += [input_var[i,0] >= -1*limits[4]]
            constraints += [input_var[i,1] >= -1*limits[5]]

            
        problem = cp.Problem(cp.Minimize(objective), constraints)
        assert problem.is_dcp()
        assert problem.is_dpp()
        
        return CvxpyLayer(problem, variables=[state_var,input_var,boundary_var], parameters=[current_state_param,current_traj_param])
            
    def convex_optimization_pass(self, traj, current_state, optimization_layer):
        
        """
        Description: This function solves the optimization dynamics using CVXPY.
        """
        #optimized_init_state = []
        optimzed_states = []
        optimized_inputs = []
        
        feasible = True
        
        for i in range(self.num_traj):
            
            try:
                states, inputs, boundary = optimization_layer(current_state.to(self.device),
                                                                traj[i,:,:].to(self.device), 
                                                                solver_args= self.solver_args)
                #optimized_init_state.append(init_state)
                optimzed_states.append(states)
                optimized_inputs.append(inputs)
            
            except Exception as e:
                logging.info("Exception Occured: %s", e)
                feasible = False
                return feasible, current_state, optimzed_states, optimized_inputs
            
        #optimized_init_state = torch.stack(optimized_init_state,dim = 0)
        optimzed_states = torch.stack(optimzed_states,dim = 0)
        optimized_inputs = torch.stack(optimized_inputs,dim = 0)
        
        return feasible, current_state, optimzed_states, optimized_inputs
    
    def generate_trajectories(self, pursuer_current_state, evader_current_state, optimization_layer_pusuer, optimization_layer_evader):
        
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
        
        feasible_p, pursuer_output_optimized_init_state, pursuer_output_optimized_states, pursuer_output_optimized_inputs = self.convex_optimization_pass(traj = pursuer_output, 
                                                                                                                                              current_state = pursuer_current_state,
                                                                                                                                              optimization_layer = optimization_layer_pusuer)
        
        feasible_e, evader_output_optimized_init_state, evader_output_optimized_states, evader_output_optimized_inputs = self.convex_optimization_pass(traj = evader_output, 
                                                                                                                                              current_state = evader_current_state,
                                                                                                                                              optimization_layer = optimization_layer_evader)

        feasible = feasible_p and feasible_e
        if not feasible:
            return feasible_p, feasible_e, None, None, None, None, None, None
        
        
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
                pursuer_BMG_matrix[i,j] = self.MSE_loss(pursuer_output_optimized_states[i,:,:2],evader_output_optimized_states_ref[j,:,:2]) # :2 for position
                
        evader_BMG_matrix = torch.zeros(self.num_traj,self.num_traj)
        for i in range(self.num_traj):
            for j in range(self.num_traj):
                evader_BMG_matrix[i,j] = -1 * self.MSE_loss(evader_output_optimized_states[i,:,:2],pursuer_output_optimized_states_ref[j,:,:2])
                
        pursuer_BMG_matrix_np = pursuer_BMG_matrix.clone().detach().numpy()
        evader_BMG_matrix_np = evader_BMG_matrix.clone().detach().numpy()
        
        logging.debug("Pursuer BMG Matrix: %s", pursuer_BMG_matrix)
        logging.debug("Pursuer BMG Matrix Shape: %s", pursuer_BMG_matrix.shape)
        logging.debug("Evader BMG Matrix: %s", evader_BMG_matrix)
        logging.debug("Evader BMG Matrix Shape: %s", evader_BMG_matrix.shape)
        
        game = nash.Game(pursuer_BMG_matrix_np, evader_BMG_matrix_np)
        equilibria = game.support_enumeration()   #lemke_howson_enumeration()
        
        def equilibria_key(item):
            if (item[0].shape[0] != self.num_traj) or (item[1].shape[0] != self.num_traj):
                logging.error("Equilibria Shape Mismatch")
                logging.error("Pursuer Equilibria Shape: %s", item[0].shape)
                logging.error("Evader Equilibria Shape: %s", item[1].shape)
                logging.error("Equilibria: %s", equilibria)
                return 0
            return sum(item[0] * (pursuer_BMG_matrix_np @ item[1]))

        sorted_equilibria = sorted(equilibria, key=equilibria_key)
        #sorted_equilibria = sorted(equilibria, key=lambda item: sum(item[0] * (pursuer_BMG_matrix_np @ item[1])))
        
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
        
        pursuer_final_controls = pursuer_output_optimized_inputs[pursuer_sol_index,:,:]
        evader_final_controls = evader_output_optimized_inputs[evader_sol_index,:,:]
        
        logging.debug("Pursuer Final States: %s", pursuer_final_states)
        logging.debug("Evader Final States: %s", evader_final_states)
        logging.debug("Pursuer Final States Shape: %s", pursuer_final_states.shape)
        logging.debug("Evader Final States Shape: %s", evader_final_states.shape)
        logging.debug("Pursuer Final Controls: %s", pursuer_final_controls)
        logging.debug("Evader Final Controls: %s", evader_final_controls)
        logging.debug("Pursuer Final Controls Shape: %s", pursuer_final_controls.shape)
        logging.debug("Evader Final Controls Shape: %s", evader_final_controls.shape)
        
        
        return feasible_p, feasible_e, pursuer_final_states, evader_final_states, pursuer_final_controls, evader_final_controls, pursuer_error, evader_error
    
    def generate_init_states(self, zero_vel = True):
        
        logging.info("Generating Initial States...")
        optimization_layer_pusuer = self.contruct_optimization_problem(self.pursuer_limits, self.bounds_type)
        optimization_layer_evader = self.contruct_optimization_problem(self.evader_limits, self.bounds_type)
        
        non_feasible = True
        while non_feasible:
            if self.enviroment == 'square':
                # Square  
                pursuer_init_state = (2*torch.rand(self.state_dim)-1).mul(torch.tensor(self.pursuer_limits[:self.state_dim])).to(self.device)        
                evader_init_state = (2*torch.rand(self.state_dim)-1).mul(torch.tensor(self.evader_limits[:self.state_dim])).to(self.device)
                
            
            elif self.enviroment == 'right_triangle':
                # RIght Triangle
                pursuer_init_state = (2*torch.rand(self.state_dim)-1).mul(torch.tensor(self.pursuer_limits[:self.state_dim])).to(self.device)        
                evader_init_state = (2*torch.rand(self.state_dim)-1).mul(torch.tensor(self.evader_limits[:self.state_dim])).to(self.device)

                r = self.pursuer_limits[0]
                pursuer_init_state[0] = random.uniform(pursuer_init_state[1],r)
                evader_init_state[0] = random.uniform(evader_init_state[1],r)
                
            elif self.enviroment == 'pentagon':
                # Pentagon
                m = np.zeros((5,))
                c = np.zeros((5,))
                
                for i in range(len(m)):
                    r = self.pursuer_limits[0]
                    m[i] = ( (np.cos((i+1)*(2*np.pi/5)) - np.cos((i)*(2*np.pi/5))) / ( np.sin((i+1)*(2*np.pi/5)) - np.sin((i)*(2*np.pi/5)) ) )
                    c[i] = r*np.cos((i)*(2*np.pi/5)) - m[i] * r * np.sin((i)*(2*np.pi/5))
                    
                pursuer_init_state = (2*torch.rand(self.state_dim)-1).mul(torch.tensor(self.pursuer_limits[:self.state_dim])).to(self.device)        
                while True:
                    
                    # the 5 lines of the pentagon
                    if (    (pursuer_init_state[1] <=  pursuer_init_state[0]*m[0] + c[0]) and
                            (pursuer_init_state[1] >=  pursuer_init_state[0]*m[1] + c[1]) and
                            (pursuer_init_state[1] >=  pursuer_init_state[0]*m[2] + c[2]) and
                            (pursuer_init_state[1] >=  pursuer_init_state[0]*m[3] + c[3]) and
                            (pursuer_init_state[1] <=  pursuer_init_state[0]*m[4] + c[4])   ):
                        
                        break
                    
                    logging.info("Pursuer Init State Not in Pentagon, Trying Again...")
                    r = random.uniform(0,5)
                    theta = random.uniform(0,2*np.pi)
                    pursuer_init_state[0] = r*np.cos(theta)
                    pursuer_init_state[1] = r*np.sin(theta)
                    
                evader_init_state = (2*torch.rand(self.state_dim)-1).mul(torch.tensor(self.evader_limits[:self.state_dim])).to(self.device)
                while True:
                    
                    # the 5 lines of the pentagon
                    if (    (evader_init_state[1] <=  evader_init_state[0]*m[0] + c[0]) and
                            (evader_init_state[1] >=  evader_init_state[0]*m[1] + c[1]) and
                            (evader_init_state[1] >=  evader_init_state[0]*m[2] + c[2]) and
                            (evader_init_state[1] >=  evader_init_state[0]*m[3] + c[3]) and
                            (evader_init_state[1] <=  evader_init_state[0]*m[4] + c[4])   ):
                        
                        break
                    
                    logging.info("Evader Init State Not in Pentagon. Trying Again...")
                    r = random.uniform(0,5)
                    theta = random.uniform(0,2*np.pi)
                    evader_init_state[0] = r*np.cos(theta)
                    evader_init_state[1] = r*np.sin(theta)
                    
                if zero_vel:
                    pursuer_init_state[2:] = 0
                    evader_init_state[2:] = 0

                feasible_p, feasible_e, pursuer_final_states, evader_final_states, pursuer_final_controls, evader_final_controls, pursuer_error, evader_error = self.generate_trajectories(pursuer_init_state, evader_init_state, optimization_layer_pusuer, optimization_layer_evader)
                if feasible_p and feasible_e:
                    non_feasible = False
                logging.info("Generated State not Feasible, Trying Again...")
        

        logging.info("Pursuer Init State: %s", pursuer_init_state)
        logging.info("Evader Init State: %s", evader_init_state)
        return pursuer_init_state, evader_init_state
    
    def train(self,P_LR, E_LR, num_epochs, save_model, load_model, reset_n_step, scheduler_step_size, scheduler_gamma):
        
        logging.info("Training Player Trajectory Generator Pytorch Model...")
        
        if load_model:
            logging.info("Loading the latest model...")
            self.load_model(get_latest_version('training','train_v'))
        
        pursuer_optimizer = optim.SGD(self.pursuer_model.parameters(), lr=P_LR) # create the pursuer optimizer
        evader_optimizer = optim.SGD(self.evader_model.parameters(), lr=E_LR)
        
        # Learning rate schedulers
        pursuer_scheduler = lr_scheduler.StepLR(pursuer_optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
        evader_scheduler = lr_scheduler.StepLR(evader_optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

        
        optimization_layer_pusuer = self.contruct_optimization_problem(self.pursuer_limits, self.bounds_type)
        optimization_layer_evader = self.contruct_optimization_problem(self.evader_limits, self.bounds_type)
        
        
        # Initialize the states_for sim
        pursuer_error_sim = np.zeros((num_epochs,1))
        evader_error_sim = np.zeros((num_epochs,1))
        pursuer_states_sim = np.zeros((num_epochs, self.state_dim))
        evader_states_sim = np.zeros((num_epochs, self.state_dim))
        pursuer_controls_sim = np.zeros((num_epochs,  self.n_steps, self.input_dim))
        evader_controls_sim = np.zeros((num_epochs,  self.n_steps, self.input_dim))
        pursuer_trajectories_sim = np.zeros((num_epochs, self.n_steps, self.state_dim))
        evader_trajectories_sim = np.zeros((num_epochs, self.n_steps, self.state_dim))
        
        
        pursuer_init_state, evader_init_state = self.generate_init_states()
        
        logging.info("Pursuer Initial State: %s", pursuer_init_state)
        logging.info("Evader Initial State: %s", evader_init_state)
        
        self.pursuer_model.train()
        self.evader_model.train()
        
        reset_counter = 0
        
        for epoch in range(num_epochs):
            
            feasible_p, feasible_e, pursuer_final_states, evader_final_states, pursuer_final_controls, evader_final_controls, pursuer_error, evader_error = self.generate_trajectories(pursuer_init_state, evader_init_state, optimization_layer_pusuer, optimization_layer_evader)
            
            if not feasible_p:

                logging.error("\n Infeasible Trajectory for Pursuer \n")
                
                
                pursuer_init_state, evader_init_state = self.generate_init_states()
                reset_counter = 0
                continue
            
            if not feasible_e:
                logging.error("\n Infeasible Trajectory for Evader \n")
                
                pursuer_init_state, evader_init_state = self.generate_init_states()
                reset_counter = 0
                continue
            
            pursuer_optimizer.zero_grad() # zero the gradients
            pursuer_error.backward()
            pursuer_optimizer.step()
            
            evader_optimizer.zero_grad() # zero the gradients
            evader_error.backward()
            evader_optimizer.step()
            
            # Update the learning rate
            pursuer_scheduler.step()
            evader_scheduler.step()
            
            # Update the states
            pursuer_error_sim[epoch] = pursuer_error.clone().detach().cpu().numpy()
            evader_error_sim[epoch] = evader_error.clone().detach().cpu().numpy()
            pursuer_states_sim[epoch,:] = pursuer_init_state.clone().detach().cpu().numpy()
            evader_states_sim[epoch,:] = evader_init_state.clone().detach().cpu().numpy()
            pursuer_trajectories_sim[epoch,:,:] = pursuer_final_states.clone().detach().cpu().numpy()
            evader_trajectories_sim[epoch,:,:] = evader_final_states.clone().detach().cpu().numpy()
            pursuer_controls_sim[epoch,:,:] = pursuer_final_controls.clone().detach().cpu().numpy()
            evader_controls_sim[epoch,:,:] = evader_final_controls.clone().detach().cpu().numpy()
            
            # logging
            logging.info("Epoch: %s / %s, Pursuer Error: %s, Evader Error: %s, Distance between pursuer and evader: %s",
                        epoch, num_epochs, pursuer_error_sim[epoch].item(), evader_error_sim[epoch].item(),
                        torch.norm(pursuer_init_state[:2] - evader_init_state[:2]).item())
            
            # check if evader caught by pursuer
            if torch.norm(pursuer_init_state[:2] - evader_init_state[:2]) < self.catch_radius:
                
                logging.info("Evader Caught by Pursuer")
                pursuer_init_state, evader_init_state = self.generate_init_states()
                reset_counter = 0
                continue

            # reset the states after 1000 epochs
            reset_counter += 1
            if reset_counter == reset_n_step:
                
                logging.info("Resetting the states")
                pursuer_init_state, evader_init_state = self.generate_init_states()
                reset_counter = 0
                continue
            
            # update the states
            pursuer_init_state = pursuer_final_states[0,:].clone().detach()
            evader_init_state = evader_final_states[0,:].clone().detach()
            

        # Save the models
        if save_model:
                      
            pursuer_model_name = self.save_path + "/pursuer_model.pth"
            evader_model_name = self.save_path + "/evader_model.pth"
            
            torch.save(self.pursuer_model.state_dict(), pursuer_model_name) 
            torch.save(self.evader_model.state_dict(), evader_model_name)
            
            # save the params
            params = {
                'num_traj': self.num_traj,
                'state_dim': self.state_dim,
                'input_dim': self.input_dim,
                'n_steps': self.n_steps,
                'dt': self.dt,
                'pursuer_limits': self.pursuer_limits,
                'evader_limits': self.evader_limits,
                'hidden_layer_num': self.hidden_layer_num,
                'solver_max_iter': self.solver_args['max_iters'],
                'device': self.device,
                'catch_radius': self.catch_radius,
                'save_path': self.save_path,
                'verbose': False,
                'solve_method': self.solver_args['solve_method'],
                'enviroment': self.enviroment,
                'margin': self.margin,
                'bounds_type': self.bounds_type
            }
            
            with open(self.save_path + '/model_params.txt', 'w') as f:
                for key, value in params.items():
                    f.write(f'{key}: {value}\n')
        
        return pursuer_error_sim, evader_error_sim, pursuer_states_sim, evader_states_sim, pursuer_controls_sim, evader_controls_sim, pursuer_trajectories_sim, evader_trajectories_sim
    
    def test(self, pursuer_init_state, evader_init_state, num_epochs):
        
        logging.info("Testing Player Trajectory Generator Pytorch Model...")
        
        optimization_layer_pusuer = self.contruct_optimization_problem(self.pursuer_limits, self.bounds_type)
        optimization_layer_evader = self.contruct_optimization_problem(self.evader_limits, self.bounds_type)
        
        logging.info("Pursuer Initial State: %s", pursuer_init_state)
        logging.info("Evader Initial State: %s", evader_init_state)
        
        
        # record the information
        pursuer_error_sim = np.zeros((num_epochs,1))
        evader_error_sim = np.zeros((num_epochs,1))
        pursuer_states_sim = np.zeros((num_epochs, self.state_dim))
        evader_states_sim = np.zeros((num_epochs, self.state_dim))
        pursuer_controls_sim = np.zeros((num_epochs,  self.n_steps, self.input_dim))
        evader_controls_sim = np.zeros((num_epochs,  self.n_steps, self.input_dim))
        pursuer_trajectories_sim = np.zeros((num_epochs, self.n_steps, self.state_dim))
        evader_trajectories_sim = np.zeros((num_epochs, self.n_steps, self.state_dim))
        distance_sim = np.zeros((num_epochs,1))
        relative_velocity_sim = np.zeros((num_epochs,1))
        relative__velocity_angle_sim = np.zeros((num_epochs,1))
        caught = False
        
        self.pursuer_model.eval()
        self.evader_model.eval()
        
        distance_sim[0] = torch.norm(pursuer_init_state[:2] - evader_init_state[:2]).item()
        relative_velocity_sim[0] = torch.norm(pursuer_init_state[2:] - evader_init_state[2:]).item()
        relative__velocity_angle_sim[0] = torch.atan2(pursuer_init_state[3] - evader_init_state[3] ,pursuer_init_state[2] - evader_init_state[2]).item()
        
        for epoch in range(num_epochs):
            
            with torch.no_grad():
                feasible_p, feasible_e,\
                pursuer_final_states, evader_final_states,\
                pursuer_final_controls, evader_final_controls,\
                pursuer_error, evader_error = self.generate_trajectories(pursuer_init_state, evader_init_state, optimization_layer_pusuer, optimization_layer_evader)
            
            feasible = feasible_p and feasible_e
            if not feasible:
                if not feasible_p:
                    logging.error("Infeasible Trajectory for Pursuer")
                if not feasible_e:
                    logging.error("Infeasible Trajectory for Evader")
                    
                # remove remaining states
                pursuer_error_sim = pursuer_error_sim[:(epoch+1),:]
                evader_error_sim = evader_error_sim[:(epoch+1),:]
                pursuer_states_sim = pursuer_states_sim[:(epoch+1),:]
                evader_states_sim = evader_states_sim[:(epoch+1),:]
                pursuer_controls_sim = pursuer_controls_sim[:(epoch+1),:,:]
                evader_controls_sim = evader_controls_sim[:(epoch+1),:,:]
                pursuer_trajectories_sim = pursuer_trajectories_sim[:(epoch+1),:,:]
                evader_trajectories_sim = evader_trajectories_sim[:(epoch+1),:,:]
                distance_sim = distance_sim[:(epoch+1)]
                relative_velocity_sim = relative_velocity_sim[:(epoch+1)]
                relative__velocity_angle_sim = relative__velocity_angle_sim[:(epoch+1)]
                
                return caught, pursuer_error_sim, evader_error_sim, pursuer_states_sim, evader_states_sim, pursuer_controls_sim, evader_controls_sim, pursuer_trajectories_sim, evader_trajectories_sim, distance_sim, relative_velocity_sim, relative__velocity_angle_sim
            
            # Update the states
            pursuer_error_sim[epoch] = pursuer_error.clone().detach().cpu().numpy()
            evader_error_sim[epoch] = evader_error.clone().detach().cpu().numpy()
            pursuer_states_sim[epoch,:] = pursuer_init_state.clone().detach().cpu().numpy()
            evader_states_sim[epoch,:] = evader_init_state.clone().detach().cpu().numpy()
            pursuer_controls_sim[epoch,:,:] = pursuer_final_controls.clone().detach().cpu().numpy()
            evader_controls_sim[epoch,:,:] = evader_final_controls.clone().detach().cpu().numpy()   
            pursuer_trajectories_sim[epoch,:,:] = pursuer_final_states.clone().detach().cpu().numpy()
            evader_trajectories_sim[epoch,:,:] = evader_final_states.clone().detach().cpu().numpy()
            distance_sim[epoch] = torch.norm(pursuer_init_state[:2] - evader_init_state[:2]).item()
            relative_velocity_sim[epoch] = torch.norm(pursuer_init_state[2:] - evader_init_state[2:]).item()
            relative__velocity_angle_sim[epoch] = torch.atan2(pursuer_init_state[3] - evader_init_state[3] ,pursuer_init_state[2] - evader_init_state[2]).item()
            
            # check if evader caught by pursuer
            pursuer_init_state = pursuer_final_states[0,:].clone().detach()
            evader_init_state = evader_final_states[0,:].clone().detach()
            
            if distance_sim[epoch] < self.catch_radius:
                
                logging.info("Evader Caught by Pursuer")
                caught = True
                # remove remaining states
                pursuer_error_sim = pursuer_error_sim[:(epoch+1),:]
                evader_error_sim = evader_error_sim[:(epoch+1),:]
                pursuer_states_sim = pursuer_states_sim[:(epoch+1),:]
                evader_states_sim = evader_states_sim[:(epoch+1),:]
                pursuer_controls_sim = pursuer_controls_sim[:(epoch+1),:,:]
                evader_controls_sim = evader_controls_sim[:(epoch+1),:,:]
                pursuer_trajectories_sim = pursuer_trajectories_sim[:(epoch+1),:,:]
                evader_trajectories_sim = evader_trajectories_sim[:(epoch+1),:,:]
                distance_sim = distance_sim[:(epoch+1)]
                relative_velocity_sim = relative_velocity_sim[:(epoch+1)]
                relative__velocity_angle_sim = relative__velocity_angle_sim[:(epoch+1)]
                
                return caught, pursuer_error_sim, evader_error_sim, pursuer_states_sim, evader_states_sim, pursuer_controls_sim, evader_controls_sim, pursuer_trajectories_sim, evader_trajectories_sim, distance_sim, relative_velocity_sim, relative__velocity_angle_sim

            
            logging.info("Epoch: %s, Pursuer Error: %s, Evader Error: %s, Distance between pursuer and evader: %s",
                        epoch, pursuer_error, evader_error,
                        torch.norm(pursuer_init_state[:2] - evader_init_state[:2]).item())
        
        return caught, pursuer_error_sim, evader_error_sim, pursuer_states_sim, evader_states_sim, pursuer_controls_sim, evader_controls_sim, pursuer_trajectories_sim, evader_trajectories_sim, distance_sim, relative_velocity_sim, relative__velocity_angle_sim
    
    def load_model(self, version):
        
        try:
            pursuer_model_name = "training/train_v" + str(version) + "/pursuer_model.pth"
            evader_model_name = "training/train_v" + str(version) + "/evader_model.pth"
        except FileNotFoundError:
            logging.error("Model not found")
            return
        
        self.pursuer_model.load_state_dict(torch.load(pursuer_model_name))
        self.evader_model.load_state_dict(torch.load(evader_model_name))
        
        return
    
    def plot_losses(self, pursuer_losses, evader_losses, name = "/losses_plot.png"):
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
        plot_name = self.save_path + name
        plt.savefig(plot_name)
        plt.gcf().patch.set_facecolor('None')
        plt.savefig(self.save_path + "/losses_plot_transparent.png",  transparent=True)
        
    def plot_data(self, distances, errors, relative_velocities, relative__velocity_angle_sim, name="/test_plot.png", figsize=(20, 10)):
        
        epochs = range(1, len(distances) + 1)

        plt.figure(figsize=figsize)
        plt.plot(epochs, distances, label='Distance')
        plt.title('Distance over Epochs')
        plt.xlabel('Epoch)')
        plt.ylabel('Distance (m)')
        plt.grid(True)
        plt.ylim(0, np.sqrt((self.pursuer_limits[0]*2)**2 + (self.pursuer_limits[1]*2)**2))

        plt.figure(figsize=figsize)
        plt.plot(epochs, errors, label='Error')
        plt.title('Pursuer Error over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Error')
        plt.grid(True)

        plt.figure(figsize=figsize)
        normalised_angles = (relative__velocity_angle_sim + np.pi) / (2 * np.pi)              
        plt.scatter(epochs,relative_velocities, c=cm.twilight_shifted(normalised_angles), edgecolor='none')
        cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='twilight_shifted'))
        cbar.set_label('Angle (rad)')
        cbar.set_ticks([0,0.5,1])
        cbar.set_ticklabels(['-π', '0', 'π'])
        plt.title('Relative Velocity [|V_p - V_e|] over Epochs]}')
        plt.xlabel('Epoch')
        plt.ylabel('Relative Velocity')
        plt.grid(True)

        plot_name = self.save_path + name 
        plt.tight_layout()
        plt.savefig(plot_name)
            
    def animate(self,pursuer_states, evader_states, pursuer_trajectories, evader_trajectories, name):
        
        plt.rcParams['animation.ffmpeg_path'] = 'C:\\Users\\Avvienash\\Documents\\ffmpeg-6.1-essentials_build\\ffmpeg-6.1-essentials_build\\bin\\ffmpeg.exe'

        fps = 15

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        
        distances = np.linalg.norm(pursuer_states[:,:2] - evader_states[:,:2], axis=1)
        distance_start_index = 0

        # Left subplot: Top-down simulation
        pursuer_pos_plot, = axs[0].plot([], [], 'ro', markersize=10)
        pursuer_traj_plot, = axs[0].plot([], [], 'r-')
        evader_pos_plot, = axs[0].plot([], [], 'bo', markersize=10)
        evader_traj_plot, = axs[0].plot([], [], 'b-')

        r = self.pursuer_limits[0]
        if self.enviroment == 'square':
            # Plot a border at xy limits of -5 to 5
            border_x = [-r, r, r, -r,-r]
            border_y = [-r, -r, r, r, -r]
            axs[0].plot(border_x, border_y, 'k-', linewidth=1)


        elif self.enviroment == 'right_triangle':
            border_x = [-r, r, r, -r]
            border_y = [-r, r, -r, -r]
            axs[0].plot(border_x, border_y, 'k-', linewidth=1)
            
        elif self.enviroment == 'pentagon':
            
            border_x = r*np.sin(np.arange(0, 2*np.pi + 2*np.pi/5, 2*np.pi/5))
            border_y = r*np.cos(np.arange(0, 2*np.pi + 2*np.pi/5, 2*np.pi/5))
            axs[0].plot(border_x, border_y, 'k-', linewidth=1)
            

        axs[0].set_xlabel('x position')
        axs[0].set_ylabel('y position')
        axs[0].set_title('Top Down Simulation')
        
        axs[0].set_xlim(-(r+0.1), (r+0.1))
        axs[0].set_ylim(-(r+0.1), (r+0.1))

        # Right subplot: Distance plot against epochs
        distance_plot, = axs[1].plot([], [], 'g-')

        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Distance')
        axs[1].set_title('Distance Plot')
        axs[1].set_ylim(0,np.sqrt((self.pursuer_limits[0]*2)**2 + (self.pursuer_limits[1]*2)**2))

        metadata = dict(title='demo', artist='avvienash')
        writer = FFMpegWriter(fps=fps, metadata=metadata)

        with writer.saving(fig, name, 100):
            for i in range(evader_states.shape[0]):

                pursuer_traj_plot_x = pursuer_trajectories[i, :, 0]
                pursuer_traj_plot_y = pursuer_trajectories[i, :, 1]
                pursuer_traj_plot.set_data(pursuer_traj_plot_x, pursuer_traj_plot_y)

                evader_traj_plot_x = evader_trajectories[i, :, 0]
                evader_traj_plot_y = evader_trajectories[i, :, 1]
                evader_traj_plot.set_data(evader_traj_plot_x, evader_traj_plot_y)

                pursuer_pos_plot_x = pursuer_states[i, 0]
                pursuer_pos_plot_y = pursuer_states[i, 1]
                pursuer_pos_plot.set_data([pursuer_pos_plot_x], [pursuer_pos_plot_y])

                evader_pos_plot_x = evader_states[i, 0]
                evader_pos_plot_y = evader_states[i, 1]
                evader_pos_plot.set_data([evader_pos_plot_x], [evader_pos_plot_y])

                distance_plot_x = np.arange(distance_start_index,(i + 1))
                distance_plot_y = distances[distance_start_index : (i + 1)]
                axs[1].set_xlim(distance_start_index,i+1)
                distance_plot.set_data(distance_plot_x, distance_plot_y)
                axs[1].set_title(f'Distance Plot: {distance_plot_y[-1]:.2f}')
                
                if distance_plot_y[-1] < self.catch_radius:
                    distance_start_index = i + 1

                logging.info("Animation Frame %s:, Distance: %s", i, distance_plot_y[-1])                  
                      
                writer.grab_frame()
        
        return 
        


def main():
    
    file_path = 'training/train_v' + str(get_latest_version('training','train') + 1) # create a new folder for training
    os.mkdir(file_path)
    
    # Set the logs file
    logs_filename = file_path + '/logs.log'
    logging_level = logging.INFO
    logging.basicConfig(level=logging_level, 
                        format='[%(levelname)s] %(message)s',
                        handlers=[ logging.StreamHandler(),logging.FileHandler(logs_filename) ])
    
    
    logging.info("File Name: %s", logs_filename)
    
    
    start_time = datetime.now() # get the start time for logging purposes
    logging.info("Start Time: %s", start_time)

    
    # Initialize the generator
    generator = PlayerTrajectoryGenerator(num_traj = 5,
                                          state_dim = 4,
                                          input_dim = 2,
                                          n_steps = 10,
                                          dt = 0.2,
                                          pursuer_limits = [2,2,0.5,0.5,0.5,0.5],
                                          evader_limits = [2,2,0.5,0.5,0.5,0.5],
                                          catch_radius = 0.1,
                                          hidden_layer_num = 100,
                                          solver_max_iter = 1000,
                                          save_path = file_path,
                                          device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                          verbose = False,
                                          solve_method = 'ECOS',
                                          enviroment = 'pentagon', # 'square', 'right_triangle', 'pentagon'
                                          margin =  0.1,
                                          bounds_type = 'hard') # create the generator
      
    logging.info("Start Training")
    logging.info("Training Parameters: %s", generator.__dict__)
    
    generator.load_model(6)
    
    pursuer_error_sim, evader_error_sim,\
    pursuer_states_sim, evader_states_sim,\
    pursuer_controls_sim, evader_controls_sim,\
    pursuer_trajectories_sim, evader_trajectories_sim = generator.train(P_LR = 0.01,
                                                                        E_LR = 0.01,
                                                                        num_epochs = 50000,
                                                                        save_model = True,
                                                                        load_model = False,
                                                                        reset_n_step = 200,
                                                                        scheduler_step_size = 500,
                                                                        scheduler_gamma = 0.95)
    
    logging.info("End Training")

    generator.plot_losses(pursuer_error_sim, evader_error_sim)

    name = file_path + "/training_animation.mp4"
    generator.animate(pursuer_states_sim, evader_states_sim, pursuer_trajectories_sim, evader_trajectories_sim, name)
    
    
    # Test the model    
    pursuer_init_state, evader_init_state = generator.generate_init_states()
    

    caught, pursuer_error_sim, evader_error_sim,\
    pursuer_states_sim, evader_states_sim,\
    pursuer_controls_sim, evader_controls_sim,\
    pursuer_trajectories_sim, evader_trajectories_sim,\
    distance_sim, relative_velocity_sim, relative__velocity_angle_sim = generator.test(pursuer_init_state,evader_init_state, num_epochs = 300)
    
    
    generator.plot_data(distance_sim,pursuer_error_sim, relative_velocity_sim, relative__velocity_angle_sim, name = "/plot_test_v1.png")
    
    name = file_path + "/testing_animation.mp4"
    generator.animate(pursuer_states_sim, evader_states_sim, pursuer_trajectories_sim, evader_trajectories_sim, name)
    
    logging.info("End Time: %s", datetime.now())
    logging.info("Time Taken: %s", datetime.now() - start_time)
    
    return
    
    
if __name__ == "__main__":
    main()
