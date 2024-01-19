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
import os


from cvxpylayers.torch import CvxpyLayer
from cvxpy.problems.objective import Maximize, Minimize
from matplotlib.animation import FFMpegWriter
from IPython.display import Video

plt.rcParams['animation.ffmpeg_path'] = 'C:\\Users\\Avvienash\\Documents\\ffmpeg-6.1-essentials_build\\ffmpeg-6.1-essentials_build\\bin\\ffmpeg.exe'
#endregion

""" Defining the Classes """

class PlayerTrajectoryGenerator(nn.Module):
    
    # Description: This class is a neural network model implemented using PyTorch's nn.Module for generating player trajectories.

    def __init__(self,num_traj,dim_x,dim_u,n_steps,xy_limit,acc_limit,input_layer_num,hidden_layer_num,output_layer_num,device):
        
        super(PlayerTrajectoryGenerator, self).__init__() # calling the parent class constructor
        
        self.num_traj = num_traj
        self.device = device
       
        # Define the Mag vector for scaling
        mag_vector = torch.ones(output_layer_num).to(self.device)
        mag_vector[0:dim_x*n_steps] = xy_limit
        mag_vector[dim_x*n_steps:] = acc_limit
        self.mag_vector = mag_vector.repeat(1,self.num_traj)
        
        # print("Mag Vector:", self.mag_vector) # all reference traj for x then all the ref for u
        # print("Mag Vector Shape:", self.mag_vector.shape)
        
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
                            self.loaded_params["dim_u"], 
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
                            self.loaded_params["dim_u"],  
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
                            self.device,
                            self.loaded_params["dim_x"],
                            self.loaded_params["n_steps"])
        
        pursuer_traj_ref = pursuer_traj.clone().detach()

        evader_traj = GetTrajFromBatchinput(
                            evader_output, 
                            evader_input, 
                            self.loaded_params["evader_num_traj"],
                            self.traj_generator,
                            self.loaded_params["solver_max_iter"],
                            self.device,
                            self.loaded_params["dim_x"],
                            self.loaded_params["n_steps"])
    
        
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
                evader_BMG_matrix[i][j] = self.MSE_loss(evader_traj[i], pursuer_traj_ref[j])

        # Solve the Bimatrix Game
        pursuer_BMG_matrix_np = pursuer_BMG_matrix.clone().detach().numpy()
        evader_BMG_matrix_np = evader_BMG_matrix.clone().detach().numpy()
        
        game = nash.Game(pursuer_BMG_matrix_np, evader_BMG_matrix_np)
        equilibria = game.vertex_enumeration() #lemke_howson_enumeration()

        def equilibria_key(item, pursuer_num_traj=self.loaded_params["pursuer_num_traj"], evader_num_traj=self.loaded_params["evader_num_traj"]):
            if (item[0].shape[0] != pursuer_num_traj) or (item[1].shape[0] != evader_num_traj):
                print("Error: Equilibria is not the correct shape")
                return 0
            return sum(item[0] * (pursuer_BMG_matrix_np @ item[1]))

        sorted_equilibria = sorted(equilibria, key=equilibria_key)
    
        pursuer_sol = torch.tensor(sorted_equilibria[0][0], dtype=torch.float)
        evader_sol = torch.tensor(sorted_equilibria[0][1], dtype=torch.float)

        # Check if the solution is the correct shape
        if evader_sol.size() != torch.Size([self.loaded_params["evader_num_traj"]]):
            new_tensor = torch.zeros(self.loaded_params["evader_num_traj"])
            new_tensor[:evader_sol.size(0)] = evader_sol
            evader_sol = new_tensor
            print("Error: Evader Solution is not the correct shape")
            
        if pursuer_sol.size() != torch.Size([self.loaded_params["pursuer_num_traj"]]):
            new_tensor = torch.zeros(self.loaded_params["pursuer_num_traj"])
            new_tensor[:pursuer_sol.size(0)] = pursuer_sol
            pursuer_sol = new_tensor
            print("Error: Pursuer Solution is not the correct shape")
            
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
        
        pursuer_states = np.zeros((max_steps, self.loaded_params["dim_x"]))
        evader_states = np.zeros((max_steps, self.loaded_params["dim_x"]))
        
        pursuer_trajectories = np.zeros((max_steps, self.loaded_params["n_steps"]*self.loaded_params["dim_x"]))
        evader_trajectories = np.zeros((max_steps, self.loaded_params["n_steps"]*self.loaded_params["dim_x"]))
        
        for frame in range(max_steps):
            
            try:
                pursuer_final_traj, evader_final_traj = self.step(pursuer_input, evader_input)
            except:
                print("Error: Trajectory Generation Failed")
                pursuer_states = pursuer_states[:frame]
                evader_states = evader_states[:frame]
                pursuer_trajectories = pursuer_trajectories[:frame]
                evader_trajectories = evader_trajectories[:frame]
                break
            
            # Store the states in the array
            pursuer_states[frame,:] = pursuer_input.cpu().clone().detach().numpy()[:self.loaded_params["dim_x"]]
            evader_states[frame,:] = evader_input.cpu().clone().detach().numpy()[:self.loaded_params["dim_x"]]
            pursuer_trajectories[frame,:] = pursuer_final_traj.cpu().clone().detach().numpy()  
            evader_trajectories[frame,:] = evader_final_traj.cpu().clone().detach().numpy()
            
            # update the states
            pursuer_input = torch.tensor([*pursuer_final_traj.clone().detach()[:self.loaded_params["dim_x"]],*evader_final_traj.clone().detach()[:self.loaded_params["dim_x"]]], dtype=torch.float)
            evader_input = torch.tensor([*evader_final_traj.clone().detach()[:self.loaded_params["dim_x"]],*pursuer_final_traj.clone().detach()[:self.loaded_params["dim_x"]]], dtype=torch.float)
            
            if (abs(evader_final_traj[0] - pursuer_final_traj[0]) <= 0.2) and  (abs(evader_final_traj[1] - pursuer_final_traj[1]) <= 0.2) :
                pursuer_states = pursuer_states[:frame+1]
                evader_states = evader_states[:frame+1]
                pursuer_trajectories = pursuer_trajectories[:frame+1]
                evader_trajectories = evader_trajectories[:frame+1]
                print("Evader Caught")
                break
            
            print("Frame:", frame)
            
        return pursuer_states, evader_states, pursuer_trajectories, evader_trajectories

""" Defining the Helper Functions """

# Function to get the latest version in a folder
def get_latest_version(folder,name):
    versions = [int(file.split('_v')[1].split('.')[0]) for file in os.listdir(folder) if file.startswith(name)]
    return max(versions) if versions else 0

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
    x = cp.Parameter(output_layer_num) # reference trajectory for states
    
    
    # Define Variables
    state_0 = cp.Variable(dim_x)
    states = cp.Variable(dim_x*T)
    controls = cp.Variable(dim_u*T)

    # Define the 
    objective = cp.norm2( W_state*10 * (state_0-x0)) # Penalises diffrence between the current state and the reference trajectory
    objective  += cp.norm2( W_state * (states-x[0:dim_x*T])) # Penalises diffrence between the current state and the reference trajectory
    objective += cp.norm2( W_control *(controls-x[dim_x*T:(dim_x+dim_u)*T])) # Penalises diffrence between the current control and the reference trajectory

    
    # Define the constraints
    constraints = [states[0:dim_x] == A_np @ state_0 + B_np @ controls[0:dim_u]] #[x0 <= xy_limit] #
    for t in range(T):
        
        if t > 0:
            constraints += [states[dim_x*t:dim_x*(t+1)] == A_np @ states[dim_x*(t-1):dim_x*(t)] + B_np @ controls[dim_u*(t):dim_u*(t+1)]]

        # xy range
        constraints += [states[dim_x*t + 0] <= xy_limit]
        constraints += [states[dim_x*t + 0] >= -xy_limit]
        constraints += [states[dim_x*t + 1] <= xy_limit]
        constraints += [states[dim_x*t + 1] >= -xy_limit]
        constraints += [states[dim_x*t + 2] <= xy_limit]
        constraints += [states[dim_x*t + 2] >= -xy_limit]
        constraints += [states[dim_x*t + 3] <= xy_limit]
        constraints += [states[dim_x*t + 3] >= -xy_limit]
        
        # # u range
        constraints += [controls[dim_u*t + 0] <=  beta]
        constraints += [controls[dim_u*t + 0] >= -beta]
        constraints += [controls[dim_u*t + 1] <=  beta]
        constraints += [controls[dim_u*t + 1] >= -beta]

        # Extra constarints
        #x + dt*vx + 0.5*dt*dt*acc_limit <= -1*xy_limit
        # constraints += [states[dim_x*t + 0] + dt*states[dim_x*t + 2] + 0.5*dt*dt*beta >= -1*xy_limit]
        # #x + dt*vx + 0.5*dt*dt*(-1*acc_limit) >= xy_limit
        # constraints += [states[dim_x*t + 0] + dt*states[dim_x*t + 2] + 0.5*dt*dt*(-1*beta) <= xy_limit]
        # # y + dt*vy + 0.5*dt*dt*acc_limit <= -1*xy_limit
        # constraints += [states[dim_x*t + 1] + dt*states[dim_x*t + 3] + 0.5*dt*dt*beta >= -1*xy_limit]
        # # y + dt*vy + 0.5*dt*dt*(-1*acc_limit) >= xy_limit
        # constraints += [states[dim_x*t + 1] + dt*states[dim_x*t + 3] + 0.5*dt*dt*(-1*beta) <= xy_limit]
        # # vx + dt*acc_limit <= -1*xy_limit
        # constraints += [states[dim_x*t + 2] + dt*beta >= -1*xy_limit]
        # # vx + dt*(-1*acc_limit) >= xy_limit
        # constraints += [states[dim_x*t + 2] + dt*(-1*beta) <= xy_limit]
        # # vy + dt*acc_limit <= -1*xy_limit
        # constraints += [states[dim_x*t + 3] + dt*beta >= -1*xy_limit]
        # # vy + dt*(-1*acc_limit) >= xy_limit
        # constraints += [states[dim_x*t + 3] + dt*(-1*beta) <= xy_limit]
        
        
    # Define the problem
    problem = cp.Problem(cp.Minimize(objective), constraints)
    assert problem.is_dcp()
    assert problem.is_dpp()
    print("is_dpp: ", problem.is_dpp())
    
    return_layer = CvxpyLayer(problem, variables=[states, controls], parameters=[x, x0])
    return return_layer

def GetTrajFromBatchinput(nn_output,nn_input,num_traj,traj_generator,solver_max_iter,device,dim_x,n_steps):

    result_list = [] # list to store the result
    solver_args = {
        "feastol": 1e-5,  # Adjust this tolerance
        # "reltol": 1e-7,   # Adjust this tolerance
        # "abstol": 1e-7, 
        "max_iters": solver_max_iter,
        "verbose": False, 
        #"acceleration_lookback": 0,
        'solve_method':'ECOS',
    }
    # iterate through the trajectories
    for i in range(num_traj):
        
            # print("nn_output:", nn_output[i])
            # print("nn_output Shape:", nn_output[i].shape)
            # print("nn_input:", nn_input[:dim_x])
            
            # print("Trajectory:", i)
            # print("nn_output:", nn_output[i])
            # print("nn_input:", nn_input[:dim_x])
            
            (current_traj_output,control) = traj_generator(nn_output[i],nn_input[:dim_x].to(device),solver_args= solver_args)
            result_list.append(current_traj_output.view(1,-1))
        
    result = torch.cat(result_list,dim=0)# concatenate the result to a tensor
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
                  
def generate_random_inputs(xy_limit, dim_x, scale_factor = 0.8):
    evader_input = torch.tensor([random.randrange(-1 * round(xy_limit * scale_factor), round(xy_limit * scale_factor)),
                                  random.randrange(-1 * round(xy_limit * scale_factor), round(xy_limit * scale_factor)),
                                  random.randrange(-1 * round(xy_limit * scale_factor), round(xy_limit * scale_factor)),
                                  random.randrange(-1 * round(xy_limit * scale_factor), round(xy_limit * scale_factor)),
                                  random.randrange(-1 * round(xy_limit * scale_factor), round(xy_limit * scale_factor)),
                                  random.randrange(-1 * round(xy_limit * scale_factor), round(xy_limit * scale_factor)),
                                  random.randrange(-1 * round(xy_limit * scale_factor), round(xy_limit * scale_factor)),
                                  random.randrange(-1 * round(xy_limit * scale_factor), round(xy_limit * scale_factor))],
                                  dtype=torch.float)

    pursuer_input = torch.cat((evader_input[dim_x:], evader_input[:dim_x]))

    return evader_input, pursuer_input       
            
def is_feasible(nn_output, nn_input, dim_x, dim_u, n_steps, xy_limit, acc_limit, dt,num_traj):
        
    state = nn_input[:dim_x].detach().cpu().numpy()
    x = state[0]
    y = state[1]
    vx = state[2]
    vy = state[3]
    
    if (x + dt*vx + 0.5*dt*dt*acc_limit <= -1*xy_limit):
        return False
    
    if (x + dt*vx + 0.5*dt*dt*(-1*acc_limit) >= xy_limit):
        return False
    
    if (y + dt*vy + 0.5*dt*dt*acc_limit <= -1*xy_limit):
        return False
    
    if (y + dt*vy + 0.5*dt*dt*(-1*acc_limit) >= xy_limit):
        return False
    
    if (vx + dt*acc_limit <= -1*xy_limit):
        return False
    
    if (vx + dt*(-1*acc_limit) >= xy_limit):
        return False
    
    if (vy + dt*acc_limit <= -1*xy_limit):
        return False
    
    if (vy + dt*(-1*acc_limit) >= xy_limit):
        return False
        
    return True
            
            
            
            
            
            
            
            
            
            
            




