#!/usr/bin/env python3

# import roslib
import sys
import os
import re
import time
import rospy
import numpy as np
import random
import roboticstoolbox as rp
import spatialmath as sm
from rv_msgs.msg import JointVelocity
from sensor_msgs.msg import JointState
import qpsolvers as qp
import matplotlib
import matplotlib.pyplot as plt
import pickle

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.style.use('ggplot')
matplotlib.rcParams['font.size'] = 4.5
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams['xtick.major.size'] = 1.5
matplotlib.rcParams['ytick.major.size'] = 1.5
matplotlib.rcParams['axes.labelpad'] = 1
plt.rc('grid', linestyle="-", color='#dbdbdb')


qdmax = np.array([
    2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100,
    5000000, 5000000, 5000000, 5000000, 5000000, 5000000])
lb = -qdmax
ub = qdmax

q0 = [-0.5653, -0.1941, -1.2602, -0.7896, -2.3227, -0.3919, -2.5173]

s0 = rp.Shape.Sphere(
    radius=0.05,
    base=sm.SE3(0.45, 0.4, 0.3)
)

s1 = rp.Shape.Sphere(
    radius=0.05,
    base=sm.SE3(0.1, 0.35, 0.65)
)

s2 = rp.Shape.Sphere(
    radius=0.02,
    base=sm.SE3(0.3, -0.3, 0)
)

s0.v = [0, -0.2, 0, 0, 0, 0]
s1.v = [0, -0.2, 0, 0, 0, 0]
s2.v = [0, 0.1, 0, 0, 0, 0]


wTe = sm.SE3(np.array([
    [ 0.99908232, -0.00979103,  0.04169728,  0.3857763 ],
    [-0.0106737,  -0.99972252,  0.02099873, -0.14162885],
    [ 0.04148011, -0.02142452, -0.9989096,   0.56963678],
    [ 0,           0,           0,           1,        ]]), check=False)


wTep = sm.SE3(np.array([
    [ 0.9987996,   0.03844678,  0.03035121,  0.50746105],
    [ 0.03559668, -0.99535677,  0.08943028,  0.43880564],
    [ 0.03364859, -0.08824252, -0.99553053,  0.30606948],
    [ 0,           0,           0,           1,        ]]), check=False)

wTs = sm.SE3(np.array([
    [ 0.49507983, -0.12231131,  0.86019527,  0.60358311],
    [-0.14425519, -0.98787197, -0.05744054, -0.07014726],
    [ 0.85678843, -0.09564998, -0.50671952,  0.84229899],
    [ 0,           0,           0,           1,        ]]), check=False)

wTinit = sm.SE3(np.array([
    [ 0.99983599, -0.01805617, -0.00140186,  0.18252277],
    [-0.01770909, -0.99094841,  0.13307002,  0.31792092],
    [-0.00379191, -0.13302337, -0.99110565,  0.49814723],
    [ 0,           0,           0,           1,        ]]), check=False)

wT_e3 = sm.SE3(
    np.array([[ 0.51728094, -0.15414056,  0.84182012,  0.4160503 ],
    [ 0.04057563, -0.97812317, -0.20403107,  0.02054094],
    [ 0.85485323,  0.13969877, -0.49971012,  0.98886652],
    [ 0.        ,  0.        ,  0.        ,  1.        ]]), check=False)

wT_e4 = sm.SE3(
    np.array([[ 0.95775419, -0.21400353, -0.19211821,  0.30808475],
    [ 0.20259281,  0.97619716, -0.07742901, -0.32218317],
    [ 0.20411533,  0.03523619,  0.9783125 ,  0.7300475 ],
    [ 0.        ,  0.        ,  0.        ,  1.        ]]), check=False)

wT_e5 = sm.SE3(
    np.array([[ 0.66463707, -0.53480389, -0.5217685 , -0.10688366],
    [-0.73521985, -0.34374666, -0.58420031, -0.32759363],
    [ 0.13307642,  0.77189574, -0.62166521,  0.5877783 ],
    [ 0.        ,  0.        ,  0.        ,  1.        ]]), check=False)

wT_e6 = sm.SE3(
    np.array([[ 0.99847581, -0.00816676, -0.05458349,  0.36802996],
    [-0.01046091, -0.99906799, -0.04187734,  0.3983214 ],
    [-0.05419062,  0.0423845 , -0.99763066,  0.10721047],
    [ 0.        ,  0.        ,  0.        ,  1.        ]]), check=False)

wT_e7 = sm.SE3(
    np.array([[ 0.99954374, -0.0274862 ,  0.01252281,  0.52575806],
    [-0.02733123, -0.99954974, -0.01238256, -0.16837214],
    [ 0.01285752,  0.01203465, -0.99984491,  0.08254964],
    [ 0.        ,  0.        ,  0.        ,  1.        ]]), check=False)

wT_e8 = sm.SE3(
    np.array([[ 0.99938713,  0.0239239 , -0.02555423,  0.39872703],
    [ 0.0240422 , -0.99970156,  0.00433234,  0.03783871],
    [-0.02544296, -0.00494406, -0.99966405,  0.10345876],
    [ 0.        ,  0.        ,  0.        ,  1.        ]]), check=False)

qs = np.array([-0.20738953, -0.0548877 , -1.24075604, -1.75235347, -0.03584731, 1.74775206, -0.4801399 ])
qinit = np.array([0.01001354, -0.94196696,  0.666929,   -2.35186282,  0.64417565,  1.5923856, 1.26946392])
qe = np.array([-0.36802377, -0.46035445,  0.00213448, -2.03919368,  0.03621091,  1.60855928, 0.42849982])
qe2 = np.array([-0.76802377, -0.46035445,  0.00213448, -2.03919368,  0.03621091,  1.60855928, 0.42849982])
q_e3 = np.array([ 0.72753865,  0.3939903 , -0.02553072, -1.72200977,  0.06517471, 2.19448239,  1.41855741])
q_e6 = np.array([ 0.57937697,  0.30853719, -0.02962658, -1.38546566,  0.011491, 1.79103358,  1.41999518])
q_e7 = np.array([ 0.58136559, -0.47102934,  0.2124923 , -2.17903279,  0.12434963, 1.77409287,  0.98258544])


class Pbvs():

    def __init__(self):

        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(2.5, 1.2)
        self.ax.set(xlabel='Time (s)', ylabel='Manipulability')
        self.ax.grid()
        plt.grid(True)
        self.ax.set_xlim(xmin=0, xmax=3.1)
        self.ax.set_ylim(ymin=0, ymax=0.11)
        plt.subplots_adjust(left=0.13, bottom=0.18, top=0.95, right=1)
        plt.ion()
        plt.show()

        # Plot the robot links
        self.rrm = self.ax.plot(
            [0], [0], label='RRMC')  #, color='#E16F6D')

        # Plot the robot links
        self.gpm = self.ax.plot(
            [0], [0], label='Park [5]')  #, color='#E16F6D')

        # Plot the robot links
        self.qua = self.ax.plot(
            [0], [0], label='MMC (ours)')  #, color='#E16F6D')

        self.ax.legend()
        self.ax.legend(loc="lower right")

        plt.pause(0.1)

        self.joint_sub = rospy.Subscriber(
            "/joint_states", 
            JointState, 
            self.state_callback)

        self.velocity_pub = rospy.Publisher(
            "/robot_driver/in/joint/velocity", 
            JointVelocity, 
            queue_size=20)

        self.start_time = time.time()
        self.data_time = []
        self.data_error = []
        self.data_feature = []

        self.r = rp.models.Panda()
        self.n = 7
        self.qlim = self.r.qlim.copy()
        self.rang = np.abs(self.qlim[0, :]) + np.abs(self.qlim[1, :])

        self.r.failt = 0
        self.r.arrivedt = 0
        self.r.s = False
        self.r.st = 0
        self.r.mt = []
        self.r.mft = []
        self.r.missed = 0

        # Timestep
        self.dt = 50
        self.ms = 0.05
        self.itmax = 100

        rospy.sleep(1)

        # self.init(qinit)
        self.relaying(qs, wTs)

    def state_callback(self, data):
        all_angles = np.array(data.position)
        self.r.q = np.r_[all_angles[0:7], 0, 0]

    def init(self, q):
        servo = True

        while not rospy.is_shutdown() and servo:
            j_vel = j_servo(self.r.q[:7], q[:7])
            self.velocity_pub.publish(j_vel)
            print(np.sum(np.abs(q[:7] - self.r.q[:7])))

            if np.sum(np.abs(q[:7] - self.r.q[:7])) < 0.15:
                servo = False

    def state(self, Ts):
        arrived = False
        eTep = self.r.fkine().inv() * Ts
        e = np.sum(np.abs(np.r_[eTep.t, eTep.rpy() * np.pi/180]))
        m = self.r.manipulability()

        if e < 0.1:
            arrived = True
        return e, m, arrived

    def step_r(self, Ts, s_time):
        e, m, _ = self.state(Ts)
        v, arrived = rp.p_servo(self.r.fkine(), Ts, 1.0, threshold=0.17)

        self.r.qd = np.linalg.pinv(self.r.jacobe()) @ v
        self.r.manip.append(m)
        self.velocity_pub.publish(np.squeeze(self.r.qd[:7]))
        c_time = time.time()
        self.r.time.append(c_time-s_time)

        self.rrm[0].set_xdata(self.r.time)
        self.rrm[0].set_ydata(self.r.manip)

        return arrived

    def step_q(self, Ts, s_time):
        ps = 0.05
        pi = 0.9

        e, m, _ = self.state(Ts)
        v, arrived = rp.p_servo(self.r.fkine(), Ts, 1.05, threshold=0.17)
        Y = 0.01

        Ain = np.zeros((self.n + 6, self.n + 6))
        bin = np.zeros(self.n + 6)

        for i in range(self.n):
            if self.r.q[i] - self.qlim[0, i] <= pi:
                bin[i] = -1.0 * (((self.qlim[0, i] - self.r.q[i]) + ps) / (pi - ps))
                Ain[i, i] = -1
            if self.qlim[1, i] - self.r.q[i] <= pi:
                bin[i] = ((self.qlim[1, i] - self.r.q[i]) - ps) / (pi - ps)
                Ain[i, i] = 1

        Q = np.eye(self.n + 6)
        Q[:self.n, :self.n] *= Y
        Q[self.n:, self.n:] = (1 / e) * np.eye(6)
        Aeq = np.c_[self.r.jacobe(), np.eye(6)]
        beq = v.reshape((6,))
        c = np.r_[-self.r.jacobm().reshape((self.n,)), np.zeros(6)]
        qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq)

        if np.any(np.isnan(qd)):
            self.r.fail = True
            self.r.s = True
            self.r.qd = self.r.qz
        else:
            self.r.qd = qd[:self.n]

        self.r.manip.append(m)

        self.velocity_pub.publish(np.squeeze(self.r.qd[:7]))

        c_time = time.time()
        self.r.time.append(c_time-s_time)
        self.qua[0].set_xdata(self.r.time)
        self.qua[0].set_ydata(self.r.manip)

        return arrived

    def step_g(self, Ts, s_time):
        e, m, _ = self.state(Ts)
        v, arrived = rp.p_servo(self.r.fkine(), Ts, 1.0, threshold=0.17)
        Y = 0.05

        Q = np.eye(self.n) * Y
        Aeq = self.r.jacobe()
        beq = v.reshape((6,))
        c = -self.r.jacobm().reshape((self.n,))
        qd = qp.solve_qp(Q, c, None, None, Aeq, beq)

        if np.any(np.isnan(qd)):
            self.r.fail = True
            self.r.s = True
            self.r.qd = self.r.qz
        else:
            self.r.qd = qd[:self.n]

        self.r.manip.append(m)
        self.velocity_pub.publish(np.squeeze(self.r.qd[:7]))
        c_time = time.time()
        self.r.time.append(c_time-s_time)

        self.gpm[0].set_xdata(self.r.time)
        self.gpm[0].set_ydata(self.r.manip)

        return arrived

    def relaying(self, qinit, Ts):

        q_init = np.r_[qinit, 0, 0]

        self.r.qd = np.zeros(self.n)
        self.r.it = 0
        self.r.manip = [self.r.manipulability()]
        self.r.time = [0.0]

        self.done = False
        self.it = 0

        rate = rospy.Rate(1000) # 100hz

        # gpm
        time.sleep(1)
        self.init(qinit)
        print('Initialised')
        time.sleep(1)
        self.r.manip = [self.r.manipulability()]
        self.r.time = [0.0]
        arrived = False
        s_time = time.time()
        while not rospy.is_shutdown() and not arrived:
            arrived = self.step_g(Ts, s_time)
            # plt.pause(0.001)
            rate.sleep()

        # Do Pseudoinverse
        time.sleep(1)
        self.init(qinit)
        time.sleep(1)
        print('Initialised')
        arrived = False
        s_time = time.time()
        while not rospy.is_shutdown() and not arrived:
            arrived = self.step_r(Ts, s_time)
            # plt.pause(0.001)
            rate.sleep()

        time.sleep(1)
        self.init(qinit)
        time.sleep(1)
        print('Initialised')
        self.r.manip = [self.r.manipulability()]
        self.r.time = [0.0]
        arrived = False
        s_time = time.time()
        while not rospy.is_shutdown() and not arrived:
            arrived = self.step_q(Ts, s_time)
            # plt.pause(0.001)
            rate.sleep()

        print('Finished')
        plt.pause(0.001)
        plt.ioff()
        plt.show()


def j_servo(cq, dq):

    Y = np.squeeze(np.ones((7,1))) * 0.5
    vel =  Y * (dq-cq)

    return vel


def main(args):

    rospy.init_node('pbvs')
    Pbvs()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main(sys.argv)
