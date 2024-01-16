import pygame #Successfully installed pygame-2.1.2
import random
from pygame.locals import *
import sys

# name : Avvienash Jaganathan
# Date Modified : 11/8/2022

# initialise pygame
pygame.init()

# Create Window for game:
WIDTH, HEIGHT = 1500,500 # define height and width of window
win = pygame.display.set_mode((WIDTH,HEIGHT)) # create window
pygame.display.set_caption("Real time simulation") # name the window

# Set Frame Rate
FPS = 60

#Colour Constants:
white = (255,255,255)
black = (0,0,0)
red= (255, 41, 65)
blue = (27, 249, 251)


def draw(win):
    win.fill(red)
    pygame.display.update()


def main():
    run = True
    clock = pygame.time.Clock()
    
   
    while run:
        clock.tick(FPS)
        
        draw(win) # draw everything
        
        # Break out of loop if pygame.QUIT
        for event in pygame.event.get(): 
            if event.type == pygame.QUIT: 
                run = False
                break
        
        
    pygame.quit()
    sys.exit()

if __name__ == '__main__': # make sure not simply run
    main()