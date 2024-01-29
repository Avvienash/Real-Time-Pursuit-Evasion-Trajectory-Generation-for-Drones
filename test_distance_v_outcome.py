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

from cvxpylayers.torch import CvxpyLayer
from cvxpy.problems.objective import Maximize, Minimize
from matplotlib.animation import FFMpegWriter
from IPython.display import Video
from datetime import datetime

from main import *

def main():
    
    file_path = 'testing/test_v' + str(get_latest_version('testing','test') + 1) # create a new folder for training
    os.mkdir(file_path)
    
    # Set the logs file
    logs_filename = file_path + '/logs.log'
    logging.basicConfig(level=logging.INFO, 
                        format='[%(levelname)s] %(message)s',
                        handlers=[ logging.StreamHandler(),logging.FileHandler(logs_filename) ])
    
    load_version  = 2
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
                                          solve_method = params['solve_method'])
                                            
     # Load the model
    generator.load_model(version=load_version)
    
   
    # define n 
    n = 100

    init_distance_array = np.zeros(n)
    caught_array = np.zeros(n)
    t_steps_array = np.zeros(n)
    
    os.mkdir(file_path + '/distance_plots')

    for i in range(100):
        
        logging.info("Test %d / %d", i+1, n)
        
        pursuer_init_state = (2*torch.rand(generator.state_dim)-1).mul(torch.tensor([5,5,0,0])).to(generator.device)        
        evader_init_state = (2*torch.rand(generator.state_dim)-1).mul(torch.tensor([5,5,0,0])).to(generator.device)
    
        caught, pursuer_states_sim, evader_states_sim, pursuer_trajectories_sim, evader_trajectories_sim, distance_sim = generator.test(pursuer_init_state,
                                                                                                                        evader_init_state,
                                                                                                                        num_epochs = 10)
        distance_plot_name = "/distance_plots/distance_plot_test_v" + str(i) + ".png"
        generator.plot_distance(distance_sim , name = distance_plot_name)
        
        init_distance_array[i] = distance_sim[0]
        caught_array[i] = caught
        t_steps_array[i] = len(distance_sim)
        
    
    # RESULTS:
    
    # PLOT THE DISTANCE 
    
    # Plotting the distribution
    plt.figure(figsize=(10, 10))
    plt.hist(init_distance_array, bins=30, density=True, alpha=0.7, color='blue')

    # Calculating mean and standard deviation
    mean_distance = np.mean(init_distance_array)
    std_dev_distance = np.std(init_distance_array)

    # Displaying mean and standard deviation on the plot
    plt.axvline(mean_distance, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_distance:.2f}')
    plt.axvline(mean_distance + std_dev_distance, color='green', linestyle='dashed', linewidth=2, label=f'Std Dev: {std_dev_distance:.2f}')
    plt.axvline(mean_distance - std_dev_distance, color='green', linestyle='dashed', linewidth=2)

    # Adding labels and title
    plt.xlabel('Distance')
    plt.ylabel('Probability Density')
    plt.title('Distribution of Distances')

    # Adding legend
    plt.legend()    
    plt.savefig(file_path + '/distance_distribution.png')
    
    
    # plot Distance Vs Caught
    # Scatter plot
    plt.figure(figsize=(10, 10))
    plt.scatter(init_distance_array[caught_array == 0], caught_array[caught_array == 0], color='blue', label='Not Caught', alpha=0.7)
    plt.scatter(init_distance_array[caught_array == 1], caught_array[caught_array == 1], color='red', label='Caught', alpha=0.7)

    # Adding labels and title
    plt.xlabel('Distance')
    plt.ylabel('Caught (0 or 1)')
    plt.title('Relationship between Distance and Caught')

    # Adding legend
    plt.legend()
    plt.savefig(file_path + '/caught_plot.png')
    
    # Scatter plot with color based on caught_array
    plt.figure(figsize=(10, 10))
    plt.scatter(t_steps_array[caught_array == 0], init_distance_array[caught_array == 0], color='blue', label='Not Caught', alpha=0.7)
    plt.scatter(t_steps_array[caught_array == 1], init_distance_array[caught_array == 1], color='red', label='Caught', alpha=0.7)

    # Adding labels and title
    plt.xlabel('Time Step')
    plt.ylabel('Distance')
    plt.title('Scatter Plot of Time Step against Distance')

    # Adding legend
    plt.legend()
    plt.savefig(file_path + '/time_step_plot.png')
        
    
    return
    

    

if __name__ == '__main__':
    main()