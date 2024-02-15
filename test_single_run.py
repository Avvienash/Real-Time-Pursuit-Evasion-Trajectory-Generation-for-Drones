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

from main import *

def main():
    
    version =  get_latest_version('testing','test') + 1
    file_path = 'testing/test_v' + str(version) # create a new folder for training

    os.mkdir(file_path)
    
    
    # Set the logs file
    logs_filename = file_path + '/logs.log'
    logging.basicConfig(level=logging.INFO, 
                        format='[%(levelname)s] %(message)s',
                        handlers=[ logging.StreamHandler(),logging.FileHandler(logs_filename) ])
    logging.info("Testing version: %s", version)
    start_time = datetime.now() # get the start time for logging purposes
    logging.info("Start Time: %s", start_time)
    
    
    load_version  = 4
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
    
    # Test the model    
    pursuer_init_state, evader_init_state = generator.generate_init_states()

    logging.info("Pursuer Initial State: %s", pursuer_init_state)
    logging.info("Evader Initial State: %s", evader_init_state)
            
    caught, pursuer_error_sim, evader_error_sim,\
    pursuer_states_sim, evader_states_sim,\
    pursuer_controls_sim, evader_controls_sim,\
    pursuer_trajectories_sim, evader_trajectories_sim,\
    distance_sim, relative_velocity_sim, relative__velocity_angle_sim = generator.test(pursuer_init_state,evader_init_state, num_epochs = 200)
    
    
    generator.plot_data(distance_sim,pursuer_error_sim, relative_velocity_sim, relative__velocity_angle_sim, name = "/plot_test_v1.png")
    
    plt.figure(figsize=(10,10))
    epochs = range(1, len(distance_sim) + 1)

    plt.plot(epochs, distance_sim)
    plt.xlabel('Epoch')
    plt.ylabel('Distance (m)')
    plt.savefig(file_path + '/distance_plot.png')
    plt.gcf().patch.set_facecolor('None')
    plt.savefig(file_path + '/distance_plot_transparent.png',  transparent=True)
    
    name = file_path + "/testing_animation.mp4"
    generator.animate(pursuer_states_sim, evader_states_sim, pursuer_trajectories_sim, evader_trajectories_sim, name)
    
    logging.info("End Time: %s", datetime.now())
    logging.info("Time Taken: %s", datetime.now() - start_time)
    
    return
    

    

if __name__ == '__main__':
    main()