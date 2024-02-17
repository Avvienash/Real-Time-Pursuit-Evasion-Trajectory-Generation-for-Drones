#!/usr/bin/env python

from crazyflie_py import Crazyswarm
import numpy as np
import sys

def main():

    Z = 1.0
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper

    if len(sys.argv) == 2:
        
        name = str(sys.argv[1])
        cf_list = [obj for obj in swarm.allcfs.crazyflies if getattr(obj, 'prefix', None) == name]
        
        if cf_list:
            cf = cf_list[0]
            
            
            cf.takeoff(targetHeight=Z, duration=2)
            timeHelper.sleep(2)
            
            pos = np.array(cf.initialPosition) + np.array([0, 0, Z])
            cf.goTo(pos, 0, 4) 
            #timeHelper.sleep(2.5)
            
            print('press button to continue...')
            swarm.input.waitUntilButtonPressed()

            cf.land(targetHeight=0.05, duration=5)
            timeHelper.sleep(5)
            
        else:
            print("Crazyflie not detected")
            return
        
        return
    
    
    allcfs = swarm.allcfs
    for cf in allcfs.crazyflies:
        cf.notifySetpointsStop()
    allcfs.takeoff(targetHeight=Z, duration=3)
    timeHelper.sleep(3)
    
    for cf in allcfs.crazyflies:
        pos = np.array(cf.initialPosition) + np.array([0, 0, Z])
        # 0.0  ,  2.0
        # 1.902113032590307  ,  0.6180339887498949
        # 1.1755705045849465  ,  -1.6180339887498947
        # -1.175570504584946  ,  -1.6180339887498951
        # -1.9021130325903073  ,  0.6180339887498945
            
        #pos = np.array([-1.9021130325903073  ,  0.6180339887498945, Z])
        cf.goTo(pos, 0, 2.0)
        #timeHelper.sleep(8.0)

    print('press button to continue...')
    swarm.input.waitUntilButtonPressed()

    allcfs.land(targetHeight=0.02, duration=2)
    timeHelper.sleep(2)


if __name__ == '__main__':
    main()