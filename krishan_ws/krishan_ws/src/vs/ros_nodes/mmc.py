#!/usr/bin/env python3

import rospy
import numpy as np
import roboticstoolbox as rp
import spatialmath as sm
from rv_msgs.msg import JointVelocity
from sensor_msgs.msg import JointState
import time
import sys


qdmax = np.array([
    2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100,
    5000000, 5000000, 5000000, 5000000, 5000000, 5000000])
lb = -qdmax
ub = qdmax

wTe = sm.SE3(np.array([
    [ 0.99908232, -0.00979103,  0.04169728,  0.3857763 ],
    [-0.0106737,  -0.99972252,  0.02099873, -0.14162885],
    [ 0.04148011, -0.02142452, -0.9989096,   0.56963678],
    [ 0,           0,           0,           1,        ]]), check=False)

# Goal
wTep = sm.SE3(np.array([
    [ 0.9987996,   0.03844678,  0.03035121,  0.50746105],
    [ 0.03559668, -0.99535677,  0.08943028,  0.43880564],
    [ 0.03364859, -0.08824252, -0.99553053,  0.30606948],
    [ 0,           0,           0,           1,        ]]), check=False)

# Initial robot joint angles
qinit = np.array([0.14383437071, -0.86973905, -0.04865976, -2.2984894, -0.05798796, np.pi/2, np.pi/4])

class MMC():

    def __init__(self):

        self.r = rp.models.Panda()
        self.n = 7
        self.qlim = self.r.qlim.copy()
        self.rang = np.abs(self.qlim[0, :]) + np.abs(self.qlim[1, :])

        # Timestep
        self.dt = 50
        self.ms = 0.05
        self.itmax = 100

        self.joint_sub = rospy.Subscriber(
            "/joint_states", 
            JointState, 
            self.state_callback)

        self.velocity_pub = rospy.Publisher(
            "/robot_driver/in/joint/velocity", 
            JointVelocity, 
            queue_size=20)

        rospy.sleep(1)

        # self.init(qinit)
        self.relaying(qinit, wTep)

    def init(self, q):
        servo = True

        while not rospy.is_shutdown() and servo:
            j_vel = j_servo(self.r.q[:7], q[:7])
            self.velocity_pub.publish(j_vel)
            print(np.sum(np.abs(q[:7] - self.r.q[:7])))

            if np.sum(np.abs(q[:7] - self.r.q[:7])) < 0.15:
                servo = False

    def state_callback(self, data):
        all_angles = np.array(data.position)
        self.r.q = all_angles[0:7]

    def step_r(self, Ts):
        v, arrived = rp.p_servo(self.r.fkine(self.r.q), Ts, 1.0, threshold=0.17)

        self.r.qd = np.linalg.pinv(self.r.jacobe(self.r.q)) @ v
        self.velocity_pub.publish(np.squeeze(self.r.qd[:7]))

        return arrived

    def relaying(self, qinit, Ts):

        q_init = np.r_[qinit, 0, 0]

        self.r.qd = np.zeros(self.n)
        self.r.it = 0

        self.done = False
        self.it = 0

        rate = rospy.Rate(1000) # 100hz

        # Do Pseudoinverse
        time.sleep(1)
        self.init(qinit)
        time.sleep(1)
        print('Initialised')
        arrived = False

        while not rospy.is_shutdown() and not arrived:
            # action = policy.get_action()
            arrived = self.step_r(Ts)
            rate.sleep()

        print('Finished')


def j_servo(cq, dq):

    Y = np.squeeze(np.ones((7,1))) * 0.5
    vel =  Y * (dq-cq)

    return vel


def main(args):

    rospy.init_node('mmc')
    MMC()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main(sys.argv)
