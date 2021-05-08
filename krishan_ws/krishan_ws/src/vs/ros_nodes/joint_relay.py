#!/usr/bin/env python

from __future__ import print_function

# import roslib
import sys
import rospy
import numpy as np
import math
import time


from geometry_msgs.msg import Twist, Vector3, Pose, Point, Quaternion
# from robotics.ros import tf_init, frame_transform
from rv_msgs.msg import JointVelocity

class Relay:

    def __init__(self):

        self.joint_pub = rospy.Publisher(
            #"/cartesian_velocity_node_controller/cartesian_velocity", 
            "/arm/joint/velocity",
            JointVelocity, 
            queue_size=20)

        self.joint_sub = rospy.Subscriber(
            "/robot_driver/in/joint/velocity", 
            JointVelocity, 
            self.vel_callback)


        self.vel = JointVelocity([0,0,0,0,0,0,0])

        self.vel_arr = np.array([0,0,0,0,0,0,0])

        self.count = 0
        self.zero_vel = True
        self.relay()



    def relay(self):

        rate = rospy.Rate(1000) # 100hz

        while not rospy.is_shutdown():
            self.count += 1
            if self.count > 400:
                print(self.count)
                self.reduce_vel()

            if not self.any_vel() and not self.zero_vel:
                self.count = 0
                self.zero_vel = True
                self.joint_pub.publish(self.vel)
            elif self.any_vel():
                self.zero_vel = False
                self.joint_pub.publish(self.vel)
            rate.sleep()



    def any_vel(self):
        return np.any(self.vel_arr)



    def reduce_vel(self):
        self.vel_arr = 0.9 * self.vel_arr

        if np.absolute(np.max(self.vel_arr)) < 0.05:
            self.vel_arr = np.array([0,0,0,0,0,0,0])
          
        self.vel = JointVelocity(self.vel_arr)



    def vel_callback(self, data):

        self.count = 0
        self.vel_arr = np.array(data.joints)
        self.vel = JointVelocity(self.vel_arr)



def main(args):

    rospy.init_node('velPub')

    r = Relay()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    main(sys.argv)
