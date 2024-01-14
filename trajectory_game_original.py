# 07/12/2023
# Author : Jaeyong Park
# USRG, KAIST


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
from cvxpylayers.torch import CvxpyLayer

import nashpy as nash

# v12
# Binary matrix game

# pip3 install cvxpylayers

# https://github.com/cvxgrp/cvxpylayers/issues/137

#  pip3 install cvxpy==1.2.0


####################################
use_previous_weight = False
previous_weight_name = "weights_013.pth"
save_learned_weight = True
save_weight_name = "weights_014.pth"

pursuer_do_backprop   = True
evader_do_backprop    = True
reset_iter            = True
reset_iter_num        = 10

# set new trajectory index
new_traj_number = 1
new_traj_count = new_traj_number

reset_evader_caught   = False
reset_evader_cornered = False
P_LR = 1e-2
E_LR = 1e-2

num_episodes = 100
total_iteration = num_episodes * reset_iter_num
time_gap = 0.001
cost_coeff = 1
solver_max_iter = 10000

pursuer_num_traj = 2
evader_num_traj = 2

pursuer_x  = 3
pursuer_y  = 4
evader_x   = 4
evader_y   = 4
pursuer_vx = 0
pursuer_vy = 0
evader_vx = 0
evader_vy = 0

dim_x = 4
dim_u = 2

#### Optimal Control Parameter
num_step = 10
input_layer_num = 8 # x1,y1, vx1, Vy1, x2,y2, vx2,vy2
xy_limit = 5
acc_limit = 0.5
dynamics_dt = 0.2


hidden_layer_num = 100
# if you change this, please also change the params in cvxpy layer!
output_layer_num = (dim_x + dim_u) * num_step

pursuer_total_output = output_layer_num * pursuer_num_traj
evader_total_output = output_layer_num * evader_num_traj
# output_layer_num = dim_x * num_step



plt.ion()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


coefficient = 1
c1 = 1/4*(math.sqrt(5)-1)
c2 = 1/4*(math.sqrt(5)+1)
s1 = 1/4*(math.sqrt(10+2*math.sqrt(5)))
s2 = 1/4*(math.sqrt(10-2*math.sqrt(5)))

def PrintTensor(input_tensor):
    # rounded_pursuer_output = torch.round(pursuer_traj.clone().detach().cpu() * 10) / 10
    # rounded_pursuer_output_np = rounded_pursuer_output.numpy()

    input_tensor = input_tensor[0:dim_x*num_step]
    input_tensor = input_tensor.reshape(-1,dim_x)
    input_tensor = input_tensor[:,0:2].reshape(1,num_step * int(dim_x / 2))
    input_tensor = torch.transpose(input_tensor.reshape(num_step,2),0,1)

    # pursuer_traj = pursuer_traj[0:dim_x*num_step]
    # pursuer_traj = pursuer_traj.reshape(-1,dim_x)
    # pursuer_traj = pursuer_traj[:,0:2].reshape(1,num_step * int(dim_x / 2))
    # pursuer_traj = torch.transpose(pursuer_traj.reshape(num_step,2),0,1).detach().cpu().numpy()

    input_tensor0= input_tensor[0]
    input_tensor1= input_tensor[1]
    # input_tensor = torch.transpose(input_tensor.reshape(num_step,2),0,1).detach().cpu().numpy()


    input_str = np.array2string(input_tensor0.clone().detach().cpu().numpy(), precision=1, separator=', ')
    print(input_str)
    input_str = np.array2string(input_tensor1.clone().detach().cpu().numpy(), precision=1, separator=', ')
    print(input_str)

def construct_mpc_problem():
    T = num_step # timestep
    dt = dynamics_dt # second
    beta = acc_limit # acc limit

    # dynamics
    A_np = torch.tensor([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])

    B_np = np.zeros([4,2])
    B_np[0,0] = dt*dt*1/2
    B_np[1,1] = dt*dt*1/2
    B_np[2,0] = dt
    B_np[3,1] = dt

    A = A_np.clone().detach()
    B = torch.tensor(B_np)
    
    weights_np = np.zeros([4,4])
    weights_np[0,0] = 1
    weights_np[1,1] = 1

    # constraint_x0 = np.zeros([dim_x*T,dim_x])
    # constraint_x0[0:4,0:4] = A


    x0 = cp.Parameter(dim_x) # current pose. x0.

    x = cp.Parameter(output_layer_num)
    # x = cp.Parameter(n*T)
    # x
    # 0~(nT-1) : x1,x2, ...  , xn
    # nT ~ (n+m)T : u1,u2,...,un

    states = cp.Variable(dim_x*T)
    # x u
    # x : x1 x2 v1 v2
    
    # controls = [cp.Variable(m) for _ in range(T)]
    controls = cp.Variable(dim_u*T)

    Q_matrix = np.eye(dim_x)
    R_matrix = np.eye(dim_u)

    # Objectives
    # x1
    # objective  = cp.quad_form((states[0:n]-x[0:n]),Q_matrix)
    # objective  = cp.norm2( 1000*(states[0:dim_x*T]-x[0:dim_x*T]))
    objective  = cp.norm2( 1000*(states-x[0:dim_x*T]))
    objective += cp.norm2(1*(controls-x[dim_x*T:(dim_x+dim_u)*T]))
    # objective += cp.quad_form((states[n*T:n*T+m]-x[n*T:n*T+m]),R_matrix)

    # u`Ru
    # objective += cp.quad_form((controls[0:m]-x[n*T:n*T+m]),R_matrix)
    
    # Constraints!

    # state equation
    # x1 = Ax0 + Bu1
    
    constraints = [states[0:dim_x] == A_np @ x0 + B_np @ controls[0:dim_u]]
    
    # xy range
    constraints += [states[0]<=xy_limit]
    constraints += [states[0]>=-xy_limit]
    constraints += [states[1]<=xy_limit]
    constraints += [states[1]>=-xy_limit]

    constraints += [controls[0] <=  beta]
    constraints += [controls[0] >= -beta]
    constraints += [controls[1] <=  beta]
    constraints += [controls[1] >= -beta]

    for t in range(1, T):
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


    problem = cp.Problem(cp.Minimize(objective), constraints)
    
    return_layer = CvxpyLayer(problem, variables=[states, controls], parameters=[x, x0])

    return return_layer

#########################################################

def PlotTraj(pursuer_traj,evader_traj,time,current_index):
    linewidth_vis = 1.0
    pursuer_traj = pursuer_traj[0:dim_x*num_step]
    pursuer_traj = pursuer_traj.reshape(-1,dim_x)
    pursuer_traj = pursuer_traj[:,0:2].reshape(1,num_step * int(dim_x / 2))
    pursuer_traj = torch.transpose(pursuer_traj.reshape(num_step,2),0,1).detach().cpu().numpy()

    evader_traj = evader_traj[0:dim_x*num_step]
    evader_traj = evader_traj.reshape(-1,dim_x)
    evader_traj = evader_traj[:,0:2].reshape(1,num_step * int(dim_x / 2) )
    evader_traj = torch.transpose(evader_traj.reshape(num_step,2),0,1).detach().cpu().numpy()

    plt.plot(pursuer_traj[0],pursuer_traj[1], '-',linewidth=linewidth_vis,color='b')
    plt.plot(pursuer_traj[0][current_index],pursuer_traj[1][current_index], marker='o',markersize='5',color='r')
    # plt.plot(pursuer_traj[0][1],pursuer_traj[1][1], marker='o',markersize='5',color='g')
    # plt.plot(pursuer_traj[0][19],pursuer_traj[1][19], marker='o',markersize='5',color='g')

    plt.plot(evader_traj[0],evader_traj[1], '-',linewidth=linewidth_vis,color='r')
    plt.plot(evader_traj[0][current_index],evader_traj[1][current_index], marker='x',markersize='5',color='r')
    # plt.plot(evader_traj[0][19],evader_traj[1][19], marker='x',markersize='5',color='g')
    print("plottraj function")
    print(pursuer_traj)

    visualize_limit = xy_limit
    plt.xlim([-visualize_limit,visualize_limit])
    plt.ylim([-visualize_limit,visualize_limit])
    plt.grid(True)
    plt.pause(time)

def PlotNNResultTraj(pursuer_traj,evader_traj):
    pursuer_traj = pursuer_traj[0:dim_x*num_step]
    pursuer_traj = pursuer_traj.reshape(-1,dim_x)
    pursuer_traj = pursuer_traj[:,0:2].reshape(1,num_step * int(dim_x / 2))
    pursuer_traj = torch.transpose(pursuer_traj.reshape(num_step,2),0,1).detach().cpu().numpy()

    evader_traj = evader_traj[0:dim_x*num_step]
    evader_traj = evader_traj.reshape(-1,dim_x)
    evader_traj = evader_traj[:,0:2].reshape(1,num_step * int(dim_x / 2) )
    evader_traj = torch.transpose(evader_traj.reshape(num_step,2),0,1).detach().cpu().numpy()

    plt.plot(pursuer_traj[0],pursuer_traj[1], '-',linewidth=0.5,color='b')
    # plt.plot(pursuer_traj[0][current_index],pursuer_traj[1][current_index], marker='o',markersize='5',color='r')
    # plt.plot(pursuer_traj[0][1],pursuer_traj[1][1], marker='o',markersize='5',color='g')
    # plt.plot(pursuer_traj[0][19],pursuer_traj[1][19], marker='o',markersize='5',color='g')

    plt.plot(evader_traj[0],evader_traj[1], '-',linewidth=0.5,color='r')


    # plt.plot(evader_traj[0][current_index],evader_traj[1][current_index], marker='x',markersize='5',color='r')

def PlotNNResultMultiTraj(pursuer_traj_set,evader_traj_set):
    # 
    for i in range(pursuer_num_traj):
        pursuer_traj_plot = pursuer_traj_set[i]

        pursuer_traj_plot = pursuer_traj_plot[0:dim_x*num_step]
        pursuer_traj_plot = pursuer_traj_plot.reshape(-1,dim_x)
        pursuer_traj_plot = pursuer_traj_plot[:,0:2].reshape(1,num_step * int(dim_x / 2))
        pursuer_traj_plot = torch.transpose(pursuer_traj_plot.reshape(num_step,2),0,1).detach().cpu().numpy()
        plt.plot(pursuer_traj_plot[0],pursuer_traj_plot[1], '-',linewidth=0.2,color='b')

    for i in range(evader_num_traj):
        evader_traj_plot = evader_traj_set[i]

        evader_traj_plot = evader_traj_plot[0:dim_x*num_step]
        evader_traj_plot = evader_traj_plot.reshape(-1,dim_x)
        evader_traj_plot = evader_traj_plot[:,0:2].reshape(1,num_step * int(dim_x / 2) )
        evader_traj_plot = torch.transpose(evader_traj_plot.reshape(num_step,2),0,1).detach().cpu().numpy()
        plt.plot(evader_traj_plot[0],evader_traj_plot[1], '-',linewidth=0.5,color='r')

    
    # plt.plot(pursuer_traj[0][current_index],pursuer_traj[1][current_index], marker='o',markersize='5',color='r')
    # plt.plot(pursuer_traj[0][1],pursuer_traj[1][1], marker='o',markersize='5',color='g')
    # plt.plot(pursuer_traj[0][19],pursuer_traj[1][19], marker='o',markersize='5',color='g')

    



    
    ### HERE
    # you should generate n path!
def GetTrajFromBatchinput(nn_output,nn_input,num_traj):
    print("GetTrajFromBatchinput")
    # result = torch.zeros((nn_output.shape[0] * dim_x * num_step),requires_grad = True)
    result_list = []

    print(nn_output.shape)
    for i in range(num_traj):
        (current_traj_output,control) = traj_generator(nn_output[i],nn_input[[0,1,2,3]].to(device),solver_args={"max_iters": solver_max_iter,"verbose": False})
        # result[i] = current_traj_output.clone()\
        print("for loop")
        print(current_traj_output.shape)
        result_list.append(current_traj_output.view(1,-1))

    result = torch.cat(result_list,dim=0)
    return result


# Trajectory Generator
traj_generator = construct_mpc_problem()
# Purser & Evader
class PlayerTrajectoryGenerator(nn.Module):

    def __init__(self, num_traj):
        super(PlayerTrajectoryGenerator, self).__init__()
        self.number_of_trajectory = num_traj
        self.layer1 = nn.Linear(in_features = input_layer_num,  out_features = hidden_layer_num)
        self.layer2 = nn.Linear(in_features = hidden_layer_num, out_features = hidden_layer_num)
        self.layer3 = nn.Linear(in_features = hidden_layer_num, out_features = hidden_layer_num)
        self.layer4 = nn.Linear(in_features = hidden_layer_num, out_features = hidden_layer_num)
        self.layer5 = nn.Linear(in_features = hidden_layer_num, out_features = output_layer_num * num_traj)
        # self.layer5 = construct_mpc_problem()
        

        self.network_result_ = torch.zeros((1,output_layer_num))
        # self.LeakyReLU = nn.LeakyReLU(0.1)
        # # self.act_func = F.tanh()

    def forward(self, x):


        # initial_pose = torch.tensor([x[0], x[1], x[2], x[3]], dtype = torch.float, requires_grad = True).to(device)

        x = x.to(device)
        x = F.tanh(self.layer1(x))
        x = F.tanh(self.layer2(x))
        x = F.tanh(self.layer3(x))
        mag_vector = torch.ones(1*output_layer_num).to(device)
        mag_vector[0:dim_x*num_step] = xy_limit
        mag_vector[dim_x*num_step:] = acc_limit
        mag_vector = mag_vector.repeat(1,self.number_of_trajectory)
        print(self.number_of_trajectory)
        print(mag_vector.shape)
        # x = F.tanh(self.layer4(x)).mul(mag_vector)
        x = F.tanh(self.layer4(x))
        x = F.tanh(self.layer5(x)).mul(mag_vector)
        # for visualization
        # self.network_result_ = x[0:dim_x*num_step].clone().detach()
        self.network_result_ = x.clone().detach()
        x = x.view(self.number_of_trajectory,-1)

        return x

    def getNetworkResult(self):
        # numpy
        return self.network_result_
    


pursuer_net = PlayerTrajectoryGenerator(pursuer_num_traj).to(device) 
evader_net = PlayerTrajectoryGenerator(evader_num_traj).to(device) 
# pursuer_net = PursuerTrajectoryGenerator().to(device)
# evader_net = EvaderTrajectoryGenerator().to(device)

if use_previous_weight:
    pursuer_net.load_state_dict(torch.load("weights/pursuer_"+previous_weight_name))
    evader_net.load_state_dict(torch.load( "weights/evader_" +previous_weight_name))

#model =  torch.load('model.pth')

# pursuer_optimizer    = optim.AdamW(pursuer_net.parameters(), lr=LR, amsgrad=True)
# evader_optimizer     = optim.AdamW(evader_net.parameters(), lr=LR, amsgrad=True)

pursuer_optimizer = optim.SGD(pursuer_net.parameters(), lr=P_LR)
evader_optimizer = optim.SGD(evader_net.parameters(), lr=E_LR)
pursuer_loss = nn.MSELoss()
evader_loss = nn.MSELoss()

MSE_loss = nn.MSELoss()


# only 1 input
pursuer_input = torch.tensor([pursuer_x,pursuer_y,pursuer_vx,pursuer_vy,evader_x,evader_y,evader_vx,evader_vy],dtype =torch.float)
evader_input  = torch.tensor([evader_x,evader_y,evader_vx,evader_vy,pursuer_x,pursuer_y,pursuer_vx,pursuer_vy],dtype =torch.float)

##
# multiple input
# pose_limit_sample = 3
# velocity_sample_limit = 1.5

# pursuer_input_np = np.random.uniform(low=[-pose_limit_sample,-pose_limit_sample,-velocity_sample_limit,-velocity_sample_limit,-pose_limit_sample,-pose_limit_sample,-velocity_sample_limit,-velocity_sample_limit], \
#                                       high=[pose_limit_sample,pose_limit_sample,velocity_sample_limit,velocity_sample_limit,pose_limit_sample,pose_limit_sample,velocity_sample_limit,velocity_sample_limit], \
#                                         size=(num_samples,dim_x*2))
# evader_input_np = pursuer_input_np[:,[4,5,6,7,0,1,2,3]]

# pursuer_input = torch.tensor(pursuer_input_np.reshape(num_samples,1,dim_x*2),dtype = torch.float)
# evader_input  = torch.tensor(evader_input_np.reshape(num_samples,1,dim_x*2), dtype = torch.float)

##

pursuer_error_history = []
evader_error_history = []
plt.figure(1)

# reference output for epoch 0
evader_output = evader_net(evader_input)
#(evader_traj_ref,control) = traj_generator(evader_output,evader_input[[0,1,2,3]].to(device),solver_args={"max_iters": solver_max_iter,"verbose": True})

print(evader_output.shape)
evader_traj_ref = GetTrajFromBatchinput(evader_output,evader_input,evader_num_traj)

is_verbose = True
current_iter_onetraj = 0

for i_episode in range(total_iteration):

    # 1. Train network
    print("1. Train network")
    pursuer_output = pursuer_net(pursuer_input)

    print("pursuer input output")
    print(pursuer_input)
    print(pursuer_output)
    # (pursuer_traj,control) = traj_generator(pursuer_output,pursuer_input[[0,1,2,3]].to(device),solver_args={"max_iters": solver_max_iter,"verbose": is_verbose})
    pursuer_output_set = GetTrajFromBatchinput(pursuer_output,pursuer_input,pursuer_num_traj)
    
    pursuer_traj_ref = pursuer_output_set.clone().detach() # store output reference for evader's learning

    evader_output = evader_net(evader_input)
    print("Evader input output")
    print(evader_input)
    print(evader_output)
    # (evader_traj,control) = traj_generator(evader_output,evader_input[[0,1,2,3]].to(device),solver_args={"max_iters": solver_max_iter,"verbose": is_verbose})
    evader_output_set = GetTrajFromBatchinput(evader_output,evader_input,evader_num_traj)
    evader_traj_ref = evader_output_set.clone().detach() # store output reference for pursuer's learning
    print(evader_output_set.shape)


    # 2. make bmg matrix
    print("2. make BMG matrix")
    pursuer_BMG_matrix = torch.zeros((pursuer_num_traj,evader_num_traj))
    # pursuer_BMG_matrix = np.zeros((pursuer_num_traj,evader_num_traj))
    for i in range(pursuer_num_traj):
        for j in range(evader_num_traj):
            pursuer_BMG_matrix[i][j] = MSE_loss(pursuer_output_set[i],evader_traj_ref[j])

    evader_BMG_matrix = torch.zeros((evader_num_traj,pursuer_num_traj))
    # evader_BMG_matrix = np.zeros((evader_num_traj,pursuer_num_traj))
    for i in range(evader_num_traj):
        for j in range(pursuer_num_traj):
            evader_BMG_matrix[i][j] = -1 * MSE_loss(evader_output_set[i],pursuer_traj_ref[j])


    # 3. solve BMG matrix!
    print("3. solve BMG matrix")
    print(pursuer_BMG_matrix.requires_grad)
    pursuer_BMG_matrix_np = pursuer_BMG_matrix.clone().detach().numpy()
    evader_BMG_matrix_np = evader_BMG_matrix.clone().detach().numpy()
    game = nash.Game(pursuer_BMG_matrix_np, evader_BMG_matrix_np)
    equilibria = game.lemke_howson_enumeration()

    sorted_equilibria = sorted(equilibria, key=lambda x: sum(x[0] * pursuer_BMG_matrix_np @ x[1]))


    # initial_pose = torch.tensor([x[0], x[1], x[2], x[3]], dtype = torch.float, requires_grad = True).to(device)

    pursuer_sol = torch.tensor(sorted_equilibria[0][0], dtype=torch.float)
    evader_sol = torch.tensor(sorted_equilibria[0][1],dtype=torch.float)

    # pursuer_error = pursuer_sol * pursuer_BMG_matrix * evader_sol
    pursuer_error = torch.mm(torch.mm(pursuer_sol.view(1,-1),pursuer_BMG_matrix),evader_sol.view(-1,1))
    # evader_error = evader_sol * evader_BMG_matrix * pursuer_sol
    evader_error = torch.mm(torch.mm(evader_sol.view(1,-1),evader_BMG_matrix),pursuer_sol.view(-1,1))
    print(pursuer_sol.shape)
    print(pursuer_BMG_matrix.shape)
    print(evader_sol.shape)
    print(pursuer_error.shape)


    # evader_error = -1 * evader_loss(evader_output_set, pursuer_traj_ref)
    # pursuer_error = pursuer_loss(pursuer_output_set, evader_traj_ref)


    # 4. Backpropagation
    print("4. Backpropatgation")
    print(pursuer_error.shape)
    
    pursuer_net.zero_grad()
    pursuer_error.backward()
    if pursuer_do_backprop:
        pursuer_optimizer.step()

    evader_net.zero_grad()
    evader_error.backward()
    if evader_do_backprop:
        evader_optimizer.step()

    pursuer_error_to_history = float(pursuer_error)
    pursuer_error_history.append(pursuer_error_to_history)
    
    evader_error_to_history = float(evader_error)
    evader_error_history.append(evader_error_to_history)

    # 5. Calculate trajectory
    print("5. Calculate Trajectory")
    print(pursuer_sol.shape)
    print(pursuer_output_set.shape)

    pursuer_final_traj = torch.mm(pursuer_sol.view(1,-1).to(device),pursuer_output_set)
    evader_final_traj = torch.mm(evader_sol.view(1,-1).to(device),evader_output_set)
    pursuer_final_traj = pursuer_final_traj.squeeze()
    evader_final_traj = evader_final_traj.squeeze()

    # re-set input
    if reset_iter:
        current_iter_onetraj = current_iter_onetraj + 1
        if current_iter_onetraj >= reset_iter_num:
            # Reset input!
            current_iter_onetraj = 0
            print("pursuer_Final_traj")
            print(pursuer_final_traj.shape)
            
            pursuer_input = torch.tensor([pursuer_final_traj[4*(new_traj_number-1)+0],pursuer_final_traj[4*(new_traj_number-1)+1], \
                                        pursuer_final_traj[4*(new_traj_number-1)+2],pursuer_final_traj[4*(new_traj_number-1)+3], \
                                        evader_final_traj[4*(new_traj_number-1)+0],evader_final_traj[4*(new_traj_number-1)+1], \
                                            evader_final_traj[4*(new_traj_number-1)+2],evader_final_traj[4*(new_traj_number-1)+3]], \
                                            dtype =torch.float)
            evader_input = torch.tensor([evader_final_traj[4*(new_traj_number-1)+0],evader_final_traj[4*(new_traj_number-1)+1], \
                                        evader_final_traj[4*(new_traj_number-1)+2],evader_final_traj[4*(new_traj_number-1)+3], \
                                        pursuer_final_traj[4*(new_traj_number-1)+0],pursuer_final_traj[4*(new_traj_number-1)+1], \
                                            pursuer_final_traj[4*(new_traj_number-1)+2],pursuer_final_traj[4*(new_traj_number-1)+3]], \
                                            dtype =torch.float)

    # Reset input in special case
    if reset_evader_caught:
        # if caught!
        if abs(evader_output[4*(new_traj_number-1)+0] - pursuer_output[4*(new_traj_number-1)+0]) <= 0.3 and \
            abs(evader_output[4*(new_traj_number-1)+1] - pursuer_output[4*(new_traj_number-1)+1]) <= 0.2:
            evader_input = torch.tensor([random.randrange(-(xy_limit-1),(xy_limit-1)),random.randrange(-(xy_limit-1),(xy_limit-1)),0,0,\
                                        pursuer_output[4*(new_traj_number-1)+0],pursuer_output[4*(new_traj_number-1)+1], \
                                            pursuer_output[4*(new_traj_number-1)+2],pursuer_output[4*(new_traj_number-1)+3]], \
                                            dtype = torch.float)
    if reset_evader_cornered:
        # if cornered!
        if abs(evader_output[4*(new_traj_number-1)+0])> (xy_limit-0.2) and \
            abs(evader_output[4*(new_traj_number-1)+1])> (xy_limit-0.2):
            evader_input = torch.tensor([random.randrange(-(xy_limit-1),(xy_limit-1)),random.randrange(-(xy_limit-1),(xy_limit-1)),0,0,\
                                        pursuer_output[4*(new_traj_number-1)+0],pursuer_output[4*(new_traj_number-1)+1], \
                                            pursuer_output[4*(new_traj_number-1)+2],pursuer_output[4*(new_traj_number-1)+3]], \
                                            dtype = torch.float)


    # PlotNNResultTraj(pursuer_net.getNetworkResult(),evader_net.getNetworkResult())
    # PlotNNResultTraj(pursuer_output,evader_output)
    PlotNNResultMultiTraj(pursuer_output,evader_output)
    # input()
    # PlotTraj(pursuer_traj,evader_traj,time_gap,new_traj_count)
    PlotTraj(pursuer_final_traj,evader_final_traj,time_gap,new_traj_count)
    print("print traj")


    print("Iteration : "+str(i_episode+1)+" / "+str(total_iteration) + ", Error : "+str(pursuer_error_to_history) + " / " + str(evader_error_to_history))
    plt.clf()


print("=================================")
print("save weights")

if save_learned_weight:
    torch.save(pursuer_net.state_dict(), "weights/pursuer_"+save_weight_name)
    torch.save(evader_net.state_dict(),   "weights/evader_"+save_weight_name)

# torch.save(pursuer_net,"pursher_net.pth")

# plt.pause(10000)
input()
# print(error_history[num_episodes-1])
print("End of the file")