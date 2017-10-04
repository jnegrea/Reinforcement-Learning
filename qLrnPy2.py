import numpy as np
import math
import gym
import tensorflow as tf

# Initialize gym environment
env = gym.make('CartPole-v0')


def run_episode(tmp_env, par_mat):
    obs = env.reset()
    total_reward = 0
    for _ in range(200):
        action = control0(par_mat, obs)
        # action = control1(par_mat, obs)
        obs, reward, done, info = tmp_env.step(action)
        total_reward += reward
        if done:
            break

    return total_reward


def control0(par_mat, obs):
    action = 0 if np.matmul(par_mat, obs) < 0 else 1
    return action


def control1(par_mat, obs):
    action = 0 if sigmoid(np.matmul(par_mat, obs)) < 0.5 else 1
    return action

def sigmoid(x):
    return 2 / (1 + math.exp(-x))




