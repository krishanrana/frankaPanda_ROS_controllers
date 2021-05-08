#!/usr/bin/env python

from __future__ import print_function

# import roslib
import sys
import rospy
import numpy as np
import math
import time

import roboticstoolbox as rtb
from roboticstoolbox.backends.Swift import Swift

from sensor_msgs.msg import JointState

class SwiftVis:

    def __init__(self):

        self.env = Swift()
        self.env.launch()

        self.r = rtb.models.Panda()
        self.env.add(self.r, readonly=True)

        self.joint_sub = rospy.Subscriber(
            "/joint_states", 
            JointState, 
            self.state_callback)


    def state_callback(self, data):
        all_angles = np.array(data.position)
        self.r.q = all_angles[0:7]
        self.env.step(0.0)


def main(args):

    rospy.init_node('SwiftVis')

    r = SwiftVis()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    main(sys.argv)
