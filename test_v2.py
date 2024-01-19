""" 
Author: Avvienash
Date: 12/1/2024
Description:
    This file is used to test the trained networks.
    The test is done by running a simulation between the pursuer and the evader.
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

from cvxpylayers.torch import CvxpyLayer
from matplotlib.animation import FFMpegWriter
from IPython.display import Video
from datetime import datetime
import os

from utils import *

plt.rcParams['animation.ffmpeg_path'] = 'C:\\Users\\Avvienash\\Documents\\ffmpeg-6.1-essentials_build\\ffmpeg-6.1-essentials_build\\bin\\ffmpeg.exe'


    

def main():
    
    # Create the controller
    version = get_latest_version('weights',"pursuer_weights_v")
    controller = Controller(version=version)
    
    # Set the initial states
    pursuer_input = torch.tensor([4, 4, 0, 0, 0, 0, 0, 0], dtype=torch.float)
    evader_input = torch.tensor([0, 0, 0, 0, 4, 4, 0, 0], dtype=torch.float)
    
    # Run the simulation
    pursuer_states, evader_states, pursuer_trajectories, evader_trajectories = controller.full_sim(pursuer_input, evader_input, max_steps=200)
    
    print("Evader States Shape: ", evader_states.shape)
    print("Pursuer States Shape: ", pursuer_states.shape)
    print("Evader Trajectories Shape: ", evader_trajectories.shape)
    print("Pursuer Trajectories Shape: ", pursuer_trajectories.shape)
    
    print("----------------------------------------------")
    print("Running Animation")
    print("----------------------------------------------")
    
    video_version = get_latest_version('videos',"demo_testing_v") + 1
    print("Video Version:", video_version)
    # Animate the simulation
    animate(fps = 10, 
            name = "videos/demo_testing_v" + str(video_version) + ".mp4", 
            pursuer_states = pursuer_states, 
            evader_states = evader_states, 
            pursuer_trajectories = pursuer_trajectories, 
            evader_trajectories = evader_trajectories)
    
    print("Animation Complete version: ", video_version)

if __name__ == "__main__":
    main()