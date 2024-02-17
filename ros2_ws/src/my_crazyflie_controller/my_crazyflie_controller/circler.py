#! /usr/bin/env python3

import rclpy
from rclpy.node import Node

from crazyflie_py import Crazyswarm
import numpy as np
from crazyflie_py import genericJoystick
import time

import threading

ON = True


def listener():
    global ON
    listen = genericJoystick.Joystick2()
    listen.waitUntilButtonPressed()
    ON = False
    

def generate_circle_points(radius, num_points):
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    z = np.full(num_points, 0.2)
    points = np.column_stack((x, y, z))
    return points



def main():
    
    listener_thread = threading.Thread(target=listener , daemon= False)
    listener_thread.start()
    
    print("Crazyflies found")
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    
    #cf = swarm.allcfs.crazyflies[0]
    cf_list = [obj for obj in swarm.allcfs.crazyflies if getattr(obj, 'prefix', None) == "/cf2"]
    if cf_list:
        cf = cf_list[0] 
    else:
        print("Crazyflie not detected")
        cf = None
        return
        
    print("Taking off")
    cf.takeoff(targetHeight = 0.2, duration=3)
    timeHelper.sleep(5)
    
    print("Go to Start Pos")
    cf.goTo(np.array([1.0, 0, 0.2]), 0, duration = 5)
    timeHelper.sleep(6)

    # Generate the array of points
    num_points = 30
    circle_points = generate_circle_points(radius = 1.1, num_points = num_points)
    
    

    while ON:
        for i in range(num_points):
            #print("Moving to point", circle_points[i, :])
            cf.goTo(circle_points[i, :], 0, duration=1)
            
            print(cf.paramTypeDict)
            timeHelper.sleep(2)
            
            if ON == False:
                break

    listener_thread.join()


    print("Prepare for landing")
    cf.goTo(np.array([1.0, 0.0, 0.2]), 0, duration = 5)
    timeHelper.sleep(6)
    
    print("Landing")
    cf.land(targetHeight=0.04, duration=3)
    timeHelper.sleep(4)

if __name__ == '__main__':
    main()
    