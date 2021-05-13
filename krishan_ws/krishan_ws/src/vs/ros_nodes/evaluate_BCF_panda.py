#!/usr/bin/env python3

import numpy as np
import scipy.signal
from prior_controller import RRMC_controller
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import pdb
import ray
import matplotlib.pyplot as plt
import wandb
import roboticstoolbox as rp

import rospy
import spatialmath as sm
from rv_msgs.msg import JointVelocity
from sensor_msgs import JointState
import time
import sys

ray.init()
wandb.login()
wandb.init(project="evaluation_manipulation_robot")

# Joint limits
qdmax = np.array([
    2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100,
    5000000, 5000000, 5000000, 5000000, 5000000, 5000000])
lb = -qdmax
ub = qdmax

# Goal
wTep = sm.SE3(np.array([
    [ 0.9987996,   0.03844678,  0.03035121,  0.50746105],
    [ 0.03559668, -0.99535677,  0.08943028,  0.43880564],
    [ 0.03364859, -0.08824252, -0.99553053,  0.30606948],
    [ 0,           0,           0,           1,        ]]), check=False)

# Initial robot joint angles
qinit = np.array([0.14383437071, -0.86973905, -0.04865976, -2.2984894, -0.05798796, np.pi/2, np.pi/4])



model_name = "pytorch_models/" + "manipulation_SEED:3_MCF_USE_ENSEMBLE:1_Sun_Mar_28_07:38:28_2021/manipulation_SEED:3_MCF_USE_ENSEMBLE:1_Sun_Mar_28_07:38:28_2021_"
prior = RRMC_controller(env)
sigma_prior = 0.5
NUM_AGENTS = 5
agents = [torch.load(model_name + str(i) + ".pth") for i in range(NUM_AGENTS)]
METHOD = "prior"



def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi, mu, std


class PandaEnv():

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
    # Set robot joint positions to desored intialisation pose
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
        # Compute joint velocities and publish to joints
        
        if METHOD == "prior":
            mu_prior = prior.compute_action()
            action = mu_prior
        elif METHOD == "policy":
            mu_policy, _, mu, std = agents[0](torch.FloatTensor(state).to("cuda"), True, False)
            action = mu_policy
        elif METHOD == "hybrid":
            mu_prior = prior.compute_action()
            ensemble_actions = ray.get([get_distr.remote(state,p) for p in agents])
            mu_ensemble, sigma_ensemble = fuse_ensembles_stochastic(ensemble_actions)
            mu_bcf, std_bcf = fuse_controllers(mu_prior, sigma_prior, mu_ensemble.cpu().numpy(), sigma_ensemble.cpu().numpy(), 0.5)
            action = mu_bcf

        #v, arrived = rp.p_servo(self.r.fkine(self.r.q), Ts, 1.0, threshold=0.17)
        #self.r.qd = np.linalg.pinv(self.r.jacobe(self.r.q)) @ v
        
        action = action * 0
        self.velocity_pub.publish(action)

        manip = self.r.manipulability()
        print(manip)

        if self.get_dist(wTep) < 0.1:
            arrived = True

        return arrived
    

    def get_dist(self, wTep):
        wTe = self.r.fkine(self.r.q)
        # Pose difference
        eTep = wTe.inv() * wTep
        # Translational velocity error
        ev = eTep.t
        # Compute distance to goal
        dist = np.sum(np.abs(ev))
        return dist

    
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


def fuse_ensembles_stochastic(ensemble_actions):
    mu = (np.sum(np.array([ensemble_actions[i][0] for i in range(NUM_AGENTS)]), axis=0))/NUM_AGENTS
    var = (np.sum(np.array([(ensemble_actions[i][1]**2 + ensemble_actions[i][0]**2)-mu**2 for i in range(NUM_AGENTS)]), axis=0))/NUM_AGENTS
    sigma = np.sqrt(var)
    return torch.from_numpy(mu), torch.from_numpy(sigma)

@ray.remote(num_gpus=1)
def get_distr(state, agent):
    state = torch.FloatTensor(state).unsqueeze(0).cuda()
    act, _, mu, std = agent(state, True, False)
    return [mu.detach().squeeze(0).cpu().numpy(), std.detach().squeeze(0).cpu().numpy()]

def fuse_controllers(prior_mu, prior_sigma, policy_mu, policy_sigma, zeta):
    # The policy mu and sigma are from the stochastic SAC output
    # The sigma from prior is fixed
    zeta2 = 1.0-zeta
    mu = (np.power(policy_sigma, 2) * zeta * prior_mu + np.power(prior_sigma,2) * zeta2 * policy_mu)/(np.power(policy_sigma,2) * zeta + np.power(prior_sigma,2) * zeta2)
    sigma = np.sqrt((np.power(prior_sigma,2) * np.power(policy_sigma,2))/(np.power(policy_sigma,2) * zeta + np.power(prior_sigma,2) * zeta2))
    return mu, sigma



def j_servo(cq, dq):
    # P controller to servo joints to given joint positions
    Y = np.squeeze(np.ones((7,1))) * 0.5
    vel =  Y * (dq-cq)
    return vel


def main(args):

    rospy.init_node('bcf_evaluation')
    MMC()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main(sys.argv)



