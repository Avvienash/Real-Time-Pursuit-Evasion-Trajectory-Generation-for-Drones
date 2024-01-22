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
    
    logs_version = get_latest_version('logs','logs')
    logs_filename = 'logs/logs_v' + str(logs_version + 1) + '.log'
    logging.basicConfig(level=logging.INFO, 
                        format='[%(levelname)s] %(message)s',
                        handlers=[ logging.StreamHandler(),logging.FileHandler(logs_filename) ])
    
    logging.info('Starting test script')
    
    #load the model and params
    load_version = get_latest_version('models','pursuer_model_v')
    params = {}
    file_path = f'models/model_params_v{load_version}.txt'
    logging.info("File Path: %s", file_path)
    
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

            params[key] = value
            
    # Now, you can access the parameters using loaded_params dictionary
    logging.info("Loaded Parameters: %s", params)
    
    
    generator = PlayerTrajectoryGenerator(num_traj = params['num_traj'],
                                          state_dim = params['state_dim'],
                                          input_dim = params['input_dim'],
                                          n_steps = params['n_steps'],
                                          dt = params['dt'],
                                          limits = ast.literal_eval(params['limits']),
                                          hidden_layer_num = params['hidden_layer_num'],
                                          solver_max_iter = params['solver_max_iter'],
                                          device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                          verbose = params['verbose'],
                                          solve_method = params['solve_method'])
                                            
     # Test the model
    generator.load_model(get_latest_version('models','pursuer_model_v'))
    
    pursuer_init_state = torch.tensor([0,0,0,0]).float()
    evader_init_state = torch.tensor([4,4,0,0]).float()
    
    pursuer_states_sim, evader_states_sim, pursuer_trajectories_sim, evader_trajectories_sim = generator.test(pursuer_init_state, evader_init_state, num_epochs = 200)
    name = "testing_animations/animation_v" + str(get_latest_version('testing_animations', "animation_v") + 1) + ".mp4"
    generator.animate(pursuer_states_sim, evader_states_sim, pursuer_trajectories_sim, evader_trajectories_sim, name)
    
    return
    

    

if __name__ == '__main__':
    main()