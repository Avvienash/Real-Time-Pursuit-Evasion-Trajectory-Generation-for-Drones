""" 
Author: Avvienash
Date: 12/1/2024
Description:
    This file is used to test the trained networks.
    The test is done by running a simulation between the pursuer and the evader in real time.
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


import pygame #Successfully installed pygame-2.1.2
import random
from pygame.locals import *
import sys



# initialise pygame
pygame.init()

# Create Window for game:
WIDTH, HEIGHT = 500,500 # define height and width of window
win = pygame.display.set_mode((WIDTH,HEIGHT)) # create window
pygame.display.set_caption("Real time simulation") # name the window

# Set Frame Rate
FPS = 60

#Colour Constants:
white = (255,255,255)
black = (0,0,0)
red= (255, 41, 65)
blue = (27, 249, 251)



def draw(win,pursuer_state,evader_state):
    win.fill(black)
    
    #get purseur and evader position in regular coordinates
    pursuer_x = pursuer_state[0].item()
    pursuer_y = pursuer_state[1].item()
    evader_x = evader_state[0].item()
    evader_y = evader_state[1].item()
    
    # Convert to pygame coordinates
    pursuer_x_pygame = int((pursuer_x+5)*(1/10)*WIDTH)
    pursuer_y_pygame = int((-1*pursuer_y+5)*(1/10)*HEIGHT)
    evader_x_pygame = int((evader_x+5)*(1/10)*WIDTH)
    evader_y_pygame = int((-1*evader_y+5)*(1/10)*HEIGHT)
    
    # Draw the pursuer and evader
    pygame.draw.circle(win, red, (pursuer_x_pygame,pursuer_y_pygame), 5)
    pygame.draw.circle(win, blue, (evader_x_pygame,evader_y_pygame), 5)
    
    pygame.display.update()


def main():
    run = True
    clock = pygame.time.Clock()
    
    version = get_latest_version('weights',"pursuer_weights_v")
    controller = Controller(version=version)
    
    print("---------------------------------------------------------------")
    print("Running Simulation in Real Time Version: ", version)
    print("---------------------------------------------------------------")
    
    # Set the initial states
    pursuer_state = torch.tensor([-1, -2, 0, 0], dtype=torch.float)
    evader_state = torch.tensor([4, 4, 0, 0], dtype=torch.float)
    pursuer_input = torch.tensor(np.concatenate((pursuer_state, evader_state)),dtype =torch.float) # initial input for the pursuer
    evader_input  = torch.tensor(np.concatenate((evader_state, pursuer_state)),dtype =torch.float)    
    
    while run:
        clock.tick(15)
        
        # Update the states
        pursuer_final_traj, evader_final_traj = controller.step(pursuer_input, evader_input)
        
        # update the states
        pursuer_input = torch.tensor([*pursuer_final_traj.clone().detach()[:4],*evader_final_traj.clone().detach()[:4]], dtype=torch.float)
        evader_input = torch.tensor([*evader_final_traj.clone().detach()[:4],*pursuer_final_traj.clone().detach()[:4]], dtype=torch.float)
            
        pursuer_state = pursuer_final_traj[:4].cpu()
        evader_state = evader_final_traj[:4].cpu()
        
        draw(win,pursuer_state,evader_state) # draw everything
        
        # Break out of loop if pygame.QUIT
        for event in pygame.event.get(): 
            if event.type == pygame.QUIT: 
                run = False
                break
        
        
    pygame.quit()
    sys.exit()

if __name__ == '__main__': # make sure not simply run
    main()