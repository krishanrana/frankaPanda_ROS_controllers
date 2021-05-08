import sys
import os
import re
import time
import rospy
import numpy as np
import roboticstoolbox as rtb
import spatialmath as sm
from rv_msgs.msg import JointVelocity
from sensor_msgs.msg import JointState

class Vel:

    def __init__(self):

        self.r = rtb.models.Panda()

        self.state_sub = rospy.Subscriber(
            "/joint_states",
            JointState,
            self.state_callback)

        self.velocity_pub = rospy.Publisher(
            "/ee_vel",
            JointVelocity,
            queue_size=20)

    def state_callback(self, data):
        all_angles = np.array(data.position)
        q = np.r_[all_angles[0:7]]
        dq = np.r_[np.array(data.velocity)[:7]]

        vel = self.r.jacobe(q) @ dq

        self.velocity_pub.publish(np.squeeze(vel))

def main(args):

    rospy.init_node('vel')
    Vel()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    main(sys.argv)
