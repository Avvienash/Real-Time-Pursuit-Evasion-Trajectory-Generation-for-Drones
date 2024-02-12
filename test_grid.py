""" 
Test script

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
import ast
import pandas as pd


from cvxpylayers.torch import CvxpyLayer
from cvxpy.problems.objective import Maximize, Minimize
from matplotlib.animation import FFMpegWriter
from matplotlib import cm
from IPython.display import Video
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D

from main import *

def main():
    
    file_path = 'testing/test_v' + str(get_latest_version('testing','test') + 1) # create a new folder for training
    os.mkdir(file_path)
    
    # Set the logs file
    logs_filename = file_path + '/logs.log'
    logging.basicConfig(level=logging.INFO, 
                        format='[%(levelname)s] %(message)s',
                        handlers=[ logging.StreamHandler(),logging.FileHandler(logs_filename) ])
    start_time = datetime.now() # get the start time for logging purposes
    logging.info("Start Time: %s", start_time)
    
    
    
    load_version  = 6
    logging.info("Loading version: %s", load_version)
    params_path = 'training/train_v' + str(load_version) + '/model_params.txt'
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
                                          save_path = file_path,
                                          device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                          verbose = params['verbose'],
                                          solve_method = params['solve_method'],
                                          enviroment= params['enviroment'],
                                          margin= params['margin'],
                                          bounds_type= params['bounds_type'])
                                
     # Load the model
    generator.load_model(version=load_version)
    
   
    # define n 
    n = 20
    x = np.linspace(-2, 2, n)
    y = np.linspace(-2, 2, n)
    
    x, y = np.meshgrid(x, y)
    x_flat = x.flatten()
    y_flat = y.flatten() 
    
    # penatagon boundary:
    m = np.zeros((5,))
    c = np.zeros((5,))
    
    for i in range(len(m)):
        r = 2
        m[i] = ( (np.cos((i+1)*(2*np.pi/5)) - np.cos((i)*(2*np.pi/5))) / ( np.sin((i+1)*(2*np.pi/5)) - np.sin((i)*(2*np.pi/5)) ) )
        c[i] = r*np.cos((i)*(2*np.pi/5)) - m[i] * r * np.sin((i)*(2*np.pi/5))
        
    # Filter points that are within the bounds of the pentagon
    valid_indices = []
    for i in range(len(x_flat)):
        
        if  (y_flat[i] <= x_flat[i] * m[0] + c[0]) and \
            (y_flat[i] >= x_flat[i] * m[1] + c[1]) and \
            (y_flat[i] >= x_flat[i] * m[2] + c[2]) and \
            (y_flat[i] >= x_flat[i] * m[3] + c[3]) and \
            (y_flat[i] <= x_flat[i] * m[4] + c[4]):
            
            valid_indices.append(i)
    
    x_flat  = x_flat[valid_indices]
    y_flat  = y_flat[valid_indices]
                    
    init_distance_array = np.zeros(len(x_flat))
    caught_array = np.zeros(len(x_flat))
    t_steps_array = np.zeros(len(x_flat)).astype(float)
    init_error_array = np.zeros(len(x_flat))
    init_relative_velocity_array = np.zeros(len(x_flat))
    init_relative_velocity_angle_array = np.zeros(len(x_flat))
    
    num_epochs = 200
    
    os.mkdir(file_path + '/distance_plots')
    for i in range(len(x_flat)):
        
        
        logging.info("Test %s / %s", i, len(x_flat))
        
        evader_init_state = torch.tensor([x_flat[i],y_flat[i],0,0], dtype=torch.float32).to(generator.device)
        pursuer_init_state = torch.tensor([0,0,0,0], dtype=torch.float32).to(generator.device)
    
        caught, pursuer_error_sim, evader_error_sim,\
        pursuer_states_sim, evader_states_sim,\
        pursuer_controls_sim, evader_controls_sim,\
        pursuer_trajectories_sim, evader_trajectories_sim,\
        distance_sim, relative_velocity_sim, relative__velocity_angle_sim = generator.test(pursuer_init_state,evader_init_state, num_epochs = num_epochs)
        
        # plot_name = "/distance_plots/distance_plot_test_v" + str(i) + ".png"
        # generator.plot_data(distance_sim,pursuer_error_sim, relative_velocity_sim, relative__velocity_angle_sim, name = plot_name, figsize=(4,8))


        init_distance_array[i] = distance_sim[0]
        caught_array[i] = caught
        init_error_array[i] = pursuer_error_sim[0]
        t_steps_array[i] = len(distance_sim)
        init_relative_velocity_array[i] = relative_velocity_sim[0]
        init_relative_velocity_angle_array[i] = relative__velocity_angle_sim[0]
    
    # SAVE THE RESULTS TO A CSV FILE
    data = {
        'x_flat': x_flat,
        'y_flat': y_flat,
        'init_distance_array': init_distance_array,
        'caught_array': caught_array,
        't_steps_array': t_steps_array,
        'init_error_array': init_error_array,
        'init_relative_velocity_array': init_relative_velocity_array,
        'init_relative_velocity_angle_array': init_relative_velocity_angle_array
    }

    df = pd.DataFrame(data)
    df.to_csv(file_path +'/data.csv', index=False)
    
    # RESULTS:
    
    """ PLOT 1: Position Vs Caught """
    
    plt.figure(figsize=(10, 10))
    border_x = r*np.sin(np.arange(0, 2*np.pi + 2*np.pi/5, 2*np.pi/5))
    border_y = r*np.cos(np.arange(0, 2*np.pi + 2*np.pi/5, 2*np.pi/5))
    plt.plot(border_x, border_y, 'k-', linewidth=1)
    
    plt.scatter(x_flat[caught_array == 0], y_flat[caught_array == 0], color='blue', label='Not Caught', alpha=0.3)
    plt.scatter(x_flat[caught_array == 1], y_flat[caught_array == 1], color='red', label='Caught', alpha=0.3)

    # Adding labels and title
    plt.xlabel('x')
    plt.ylabel('y')

    # Adding legend
    plt.legend(fancybox=True, framealpha=0.5)
    
    plt.savefig(file_path + '/game_outcome_plot.png')
    plt.gcf().patch.set_facecolor('None')
    plt.savefig(file_path + '/game_outcome_plot_transparent.png',  transparent=True)
    
    """ PLOT 3: Position vs catch time 2  """
    
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(x_flat, y_flat, c=cm.cool(t_steps_array/np.max(t_steps_array)), edgecolor='none', alpha=0.7,)
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='cool'))
    cbar.set_label('Catch Time')
    cbar.set_ticks([0,1])
    cbar.set_ticklabels(['0', str(np.max(t_steps_array))])
    
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.savefig(file_path + '/catch_time_plot.png')
    plt.gcf().patch.set_facecolor('None')  # Set figure face color to None for transparency
    plt.savefig(file_path + '/catch_time_plot_transparent.png', transparent=True)
    
    
    """ PLOT 2: Position vs catch time """

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Catch Time')
    surf = ax.plot_trisurf(x_flat, y_flat, t_steps_array/5, cmap=cm.cool, alpha=0.8,linewidth=0.1,vmin=0, vmax=num_epochs)
    
    plt.savefig(file_path + '/3d_surface_plot.png')
    plt.gcf().patch.set_facecolor('None')  # Set figure face color to None for transparency
    plt.savefig(file_path + '/3d_surface_plot_transparent.png', transparent=True)
    
    """ PLOT 4: Initial Distance vs Catch Time """
    
    plt.figure(figsize=(10, 10))    
    plt.scatter(init_distance_array[caught_array == 0], t_steps_array[caught_array == 0], color='blue', label='Not Caught', alpha=0.3)
    plt.scatter(init_distance_array[caught_array == 1], t_steps_array[caught_array == 1], color='red', label='Caught', alpha=0.3)

    # Adding labels and title
    plt.xlabel('Initial Distance')
    plt.ylabel('Catch Time')

    # Adding legend
    plt.legend(fancybox=True, framealpha=0.5)
    
    plt.savefig(file_path + '/distance_plot.png')
    plt.gcf().patch.set_facecolor('None')
    plt.savefig(file_path + '/distance_plot_transparent.png',  transparent=True)
    

    return
    

    

if __name__ == '__main__':
    main()