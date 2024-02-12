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
from matplotlib import cm
from IPython.display import Video
from datetime import datetime
import pandas as pd


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
    
    
    
    load_version  = 7
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
    n = 100

    init_distance_array = np.zeros(n)
    caught_array = np.zeros(n)
    t_steps_array = np.zeros(n)
    init_error_array = np.zeros(n)
    init_relative_velocity_array = np.zeros(n)
    init_relative_velocity_angle_array = np.zeros(n)
    
    os.mkdir(file_path + '/distance_plots')

    for i in range(n):
        
        logging.info("Test %d / %d", i+1, n)
        
        pursuer_init_state, evader_init_state = generator.generate_init_states()
    

        caught, pursuer_error_sim, evader_error_sim,\
        pursuer_states_sim, evader_states_sim,\
        pursuer_controls_sim, evader_controls_sim,\
        pursuer_trajectories_sim, evader_trajectories_sim,\
        distance_sim, relative_velocity_sim, relative__velocity_angle_sim = generator.test(pursuer_init_state,evader_init_state, num_epochs = 300)
        
        plot_name = "/distance_plots/distance_plot_test_v" + str(i) + ".png"
        generator.plot_data(distance_sim,pursuer_error_sim, relative_velocity_sim, relative__velocity_angle_sim, name = plot_name, figsize=(5,10))


        init_distance_array[i] = distance_sim[0]
        caught_array[i] = caught
        init_error_array[i] = pursuer_error_sim[0]
        t_steps_array[i] = len(distance_sim)
        init_relative_velocity_array[i] = relative_velocity_sim[0]
        init_relative_velocity_angle_array[i] = relative__velocity_angle_sim[0]
        
    
    # RESULTS:
    
    """ PLOT 1: DISTANCE DISTRIBUTION """
    
    # # Plotting the distribution
    # plt.figure(figsize=(10, 10))
    # plt.hist(init_distance_array, bins=30, density=True, alpha=0.7, color='blue')

    # # Calculating mean and standard deviation
    # mean_distance = np.mean(init_distance_array)
    # std_dev_distance = np.std(init_distance_array)

    # # Displaying mean and standard deviation on the plot
    # plt.axvline(mean_distance, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_distance:.2f}')
    # plt.axvline(mean_distance + std_dev_distance, color='green', linestyle='dashed', linewidth=2, label=f'Std Dev: {std_dev_distance:.2f}')
    # plt.axvline(mean_distance - std_dev_distance, color='green', linestyle='dashed', linewidth=2)

    # # Adding labels and title
    # plt.xlabel('Distance')
    # plt.ylabel('Probability Density')
    # plt.title('Distribution of Distances')

    # # Adding legend
    # plt.legend()    
    # plt.savefig(file_path + '/distance_distribution.png')
    
    
    """PLOT 2: ERROR DISTRIBUTION"""
    
    # # Plotting the distribution
    # plt.figure(figsize=(10, 10))
    # plt.hist(init_error_array, bins=30, density=True, alpha=0.7, color='blue')

    # # Calculating mean and standard deviation
    # mean_error = np.mean(init_error_array)
    # std_dev_error = np.std(init_error_array)

    # # Displaying mean and standard deviation on the plot
    # plt.axvline(mean_error, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_error:.2f}')
    # plt.axvline(mean_error + std_dev_error, color='green', linestyle='dashed', linewidth=2, label=f'Std Dev: {std_dev_error:.2f}')
    # plt.axvline(mean_error - std_dev_error, color='green', linestyle='dashed', linewidth=2)

    # # Adding labels and title
    # plt.xlabel('Error')
    # plt.ylabel('Probability Density')
    # plt.title('Distribution of Error')

    # # Adding legend
    # plt.legend()    
    # plt.savefig(file_path + '/error_distribution.png')
    
    
    """ PLOT 3: RELATIVE VELOCITY DISTRIBUTION """
    
    # # Plotting the distribution
    # plt.figure(figsize=(10, 10))
    # plt.hist(init_relative_velocity_array, bins=30, density=True, alpha=0.7, color='blue')

    # # Calculating mean and standard deviation
    # mean_relative_velocity = np.mean(init_relative_velocity_array)
    # std_dev_relative_velocity = np.std(init_relative_velocity_array)

    # # Displaying mean and standard deviation on the plot
    # plt.axvline(mean_relative_velocity, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_relative_velocity:.2f}')
    # plt.axvline(mean_relative_velocity + std_dev_relative_velocity, color='green', linestyle='dashed', linewidth=2, label=f'Std Dev: {std_dev_relative_velocity:.2f}')
    # plt.axvline(mean_relative_velocity - std_dev_relative_velocity, color='green', linestyle='dashed', linewidth=2)

    # # Adding labels and title
    # plt.xlabel('relative_velocity')
    # plt.ylabel('Probability Density')
    # plt.title('Distribution of relative_velocity')

    # # Adding legend
    # plt.legend()    
    # plt.savefig(file_path + '/relative_velocity_distribution.png')
    

    """ PLOT 4: Distance Vs Caught Time"""
    # PLOT THE TIME TAKEN TO GET CAUGHT
    
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
    
    
    # # PLOT THE TIME TAKEN TO GET CAUGHT ZOOMED IN
    # plt.figure(figsize=(10, 10))
    # plt.scatter(init_distance_array[caught_array == 1], t_steps_array[caught_array == 1], color='red', label='Caught', alpha=0.7)

    # # Adding labels and title
    # plt.xlabel('Initial distance')
    # plt.ylabel('Time Taken to get Caught')
    # plt.title('Relationship between Time Taken to get Caught and Initial Distance')

    # # Adding legend
    # plt.savefig(file_path + '/distnace_time_plot_zoomed.png')
    
    """ PLOT 5: Error vs Caught Time """
    
    # # PLOT THE TIME TAKEN TO GET CAUGHT
    
    # plt.figure(figsize=(10, 10))
    # plt.scatter(init_error_array[caught_array == 0], t_steps_array[caught_array == 0], color='blue', label='Not Caught', alpha=0.7)
    # plt.scatter(init_error_array[caught_array == 1], t_steps_array[caught_array == 1], color='red', label='Caught', alpha=0.7)

    # # Adding labels and title
    # plt.xlabel('Initial error')
    # plt.ylabel('Time Taken to get Caught')
    # plt.title('Relationship between Time Taken to get Caught and Initial error')

    # # Adding legend
    # plt.legend()
    # plt.savefig(file_path + '/error_time_plot.png')
    
    
    # # PLOT THE TIME TAKEN TO GET CAUGHT ZOOMED IN
    # plt.figure(figsize=(10, 10))
    # plt.scatter(init_error_array[caught_array == 1], t_steps_array[caught_array == 1], color='red', label='Caught', alpha=0.7)

    # # Adding labels and title
    # plt.xlabel('Initial error')
    # plt.ylabel('Time Taken to get Caught')
    # plt.title('Relationship between Time Taken to get Caught and Initial error')

    # # Adding legend
    # plt.savefig(file_path + '/error_time_plot_zoomed.png')
    
    """ PLOT 6: Relative Velocity vs Caught Time """
    
    plt.figure(figsize=(10, 10))    
    plt.scatter(init_relative_velocity_array[caught_array == 0], t_steps_array[caught_array == 0], color='blue', label='Not Caught', alpha=0.3)
    plt.scatter(init_relative_velocity_array[caught_array == 1], t_steps_array[caught_array == 1], color='red', label='Caught', alpha=0.3)

    # Adding labels and title
    plt.xlabel('Initial Relative Velocity')
    plt.ylabel('Catch Time')

    # Adding legend
    plt.legend(fancybox=True, framealpha=0.5)
    
    plt.savefig(file_path + '/velocity_plot.png')
    plt.gcf().patch.set_facecolor('None')
    plt.savefig(file_path + '/velocity_plot_transparent.png',  transparent=True)
    
    
    """ PLOT 7: KIV """
    
    
        
    return
    

    

if __name__ == '__main__':
    main()