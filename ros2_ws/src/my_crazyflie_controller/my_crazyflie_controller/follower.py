#! /usr/bin/env python3

import rclpy
from rclpy.node import Node
from motion_capture_tracking_interfaces.msg import NamedPoseArray
from crazyflie_interfaces.msg import LogDataGeneric
from crazyflie_py import genericJoystick

from crazyflie_py import Crazyswarm
import numpy as np

import threading
import time



#Global Variable
POSES = {}
VBAT = 4
listen_ON = True
pose_ON = True

# Create a lock object
lock = threading.Lock()

def listener():
    global listen_ON
    listen = genericJoystick.Joystick2()
    listen.waitUntilButtonPressed()
    with lock:
        listen_ON = False
    print("Closing Listener Thread")


class Subscriber(Node):
    
    def __init__(self):
        
        # Initialiaze The Node for th Follower
        super().__init__("follower")
        self.get_logger().info("Follower Node has started")
        
        # Poses Subscriber
        self.pose_subscriber = self.create_subscription(
                                NamedPoseArray,
                                "/poses",
                                self.pose_callback,
                                qos_profile=rclpy.qos.qos_profile_sensor_data)
        self.pose_dict = {}       
        
        self.vbat_subscription = self.create_subscription(
                                LogDataGeneric,  
                                'cf2/vbat',
                                self.vbat_callback,
                                10)
        
        self.create_timer(2,self.log_battery)
        
        self.vbat = 4
    
    def log_battery(self):
        global VBAT
        with lock:
                VBAT = self.vbat
        if self.vbat < 3.8:
            log_message = f"Battery Cirtical Volatage, Abort Mission :{self.vbat:.2f} V"
            self.get_logger().warn(log_message)
            return
        log_message = f"Battery Voltage: {self.vbat:.2f} V"
        self.get_logger().info(log_message)
        
        
        
    def vbat_callback(self, msg: LogDataGeneric):
        if msg.values[0]:
            self.vbat = msg.values[0]
            
            

    def pose_callback(self, msg: NamedPoseArray):
        global POSES
        self.pose_dict = {}

        for named_pose in msg.poses:
            name = named_pose.name
            pose = named_pose.pose.position
            position = (pose.x, pose.y, pose.z)
            self.pose_dict[name] = position

        with lock:
            POSES = self.pose_dict
        # if self.pose_dict:
        #     # Process the dictionary of pose names and positions
        #     print("Received poses:")
        #     for name, position in self.pose_dict.items():
        #         print(f"Name: {name}, Position: {position}")
        # else:
        #     print("No valid poses received")
        

def subscriber_thread_callback(args=None):
    rclpy.init(args=args)
    node = Subscriber()
    while rclpy.ok() and pose_ON:
        try:
            rclpy.spin_once(node)
        except Exception as e:
            print(f"Exception in spin_once: {e}")
        time.sleep(0.05)
    
    print("Closing Pose Subcriber Thread")
    rclpy.shutdown()        


def main():
    
    subscriber_thread = threading.Thread(target= subscriber_thread_callback , daemon= True)
    subscriber_thread.start()
    
    listener_thread = threading.Thread(target=listener , daemon= True)
    listener_thread.start()
    
    
    # Initialise Crazyswarm()
    swarm = Crazyswarm()
    print("Crazyflies found")
    timeHelper = swarm.timeHelper

    # Give time for subcriber ti tun
    time.sleep(2)
    
    # Check Battery
    print("Checking Battery Level: ", VBAT)
    if VBAT < 3.7:
        print("Critical Battery Level")
        print("Abort Mission")
        return
    elif VBAT < 3.8:
        print("Low battery Warning")
    
    #cf = swarm.allcfs.crazyflies[0]
    cf_list = [obj for obj in swarm.allcfs.crazyflies if getattr(obj, 'prefix', None) == "/cf2"]
    if cf_list:
        print("CF2 Found")
        cf = cf_list[0] 
        
    else:
        print("Crazyflie not detected")
        cf = None
        return
    
    # Take off
    cf.takeoff(targetHeight = 0.5, duration=3)
    print("Taking off")
    time.sleep(3)
    
    # Go to object
    time_step  = 0.2
    avg_speed = 0.25
    min_duration = 0.5
    target = 'cf5'
    name = 'cf2'
    
    while listen_ON:
        if (target in POSES) and (name in POSES):
            pos = np.array(POSES[target]) + np.array([0,0,0.5])
            distance = np.linalg.norm(pos - np.array(POSES[name]))
            duration = distance/avg_speed
            #print("Distance: " ,distance, "Duration :", duration)
            if distance > 0.03:
                cf.goTo(pos, 0, duration = max(duration,min_duration))
        else:
            print("cf not detected")
        time.sleep(time_step)
        
    # Stop the crazyflie
    listener_thread.join()
    global pose_ON
    pose_ON = False   
    subscriber_thread.join()
        
    # Land
    print("Prepare for landing")
    cf.goTo(np.array([0, 0, 0.5]), 0, duration = 6)
    time.sleep(6)
    
    print("Landing")
    cf.land(targetHeight=0.04, duration=3)
    time.sleep(3)

    
    try:
        rclpy.shutdown()
    except:
        print("RCLPY Already Shutdown")
    
    

if __name__ == '__main__':
    main()