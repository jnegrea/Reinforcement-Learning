import numpy as np
import tensorflow as tf
import gym
import random
import sys

import math
import matplotlib.pyplot as plt
import time


# Define Functions
def init_value_function():
    with tf.variable_scope('value_function'):
        # Defining NN Structure for value function

        # Layer 0
        state = tf.placeholder("float", [None, 4])  # State is a 1x4 Vector
        action = tf.placeholder("float", [None, 1])  # Action is a 1x1 Bool

        # Layer 1
        w1 = tf.get_variable("w1", [4, 10])  # First Layer Matrix
        b1 = tf.get_variable("b1", [10])  # First Layer Intercept
        a_w1 = tf.get_variable("a_w1", [1, 10])
        h1 = tf.nn.relu(b1 + tf.matmul(state, w1) + tf.matmul(action, a_w1))

        # Layer 2
        w2 = tf.get_variable("w2", [10, 1])  # Map back to R^1
        b2 = tf.get_variable("b2", [1])
        model_value_fun = b2 + tf.matmul(h1, w2)  # No Activation at the End

        # Loss Function
        sim_value_fun = tf.placeholder("float", [None, 1])
        diffs = sim_value_fun - model_value_fun

        loss = tf.nn.l2_loss(diffs)

        # Define Optimizer
        optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
        # optimizer = tf.train.AdagradOptimizer(0.005).minimize(loss)

        # Return
        return optimizer, state, action, model_value_fun, sim_value_fun, loss


def batch_format(batch, tf_sess, tf_model, action_n):
    # last observation, action, next observation, reward
    state0, action0, state1, reward0, done1 = (np.stack(batch[:, i]) for i in range(batch.shape[1]))

    # Get Batch Size
    batch_n = state0.shape[0]

    # Generate sims
    dummy_control = np.ones([batch_n, 1])
    q_sims = [tf_sess.run(tf_model,
                          feed_dict={vl_state: state1,
                                     vl_action: dummy_control * a}) for a in range(action_n)]

    q_sims = np.stack(q_sims, axis=1).max(axis=1)
    q_sims *= 1 - done1.reshape([batch_n, 1])
    q_sims += reward0.reshape([batch_n, 1])

    # Reshape action vector
    action0 = action0.reshape([batch_n, 1])

    return state0, action0, q_sims

# Initialize gym environment and monitor
env = gym.make('CartPole-v1')

# Set Simulation & Learning Parameters
num_episodes = 100
num_time_steps = 10**4

num_mini_batch = 64

gamma = 1
epsilon = 0.01

# Initialize Array of Transitions
transition_array = []  # Array containing Dicts

# Initialize TensorFlow Session
tf_value_fun = init_value_function()
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# Get TF objects
vl_optim, vl_state, vl_action, vl_model, vl_sim, vl_loss = tf_value_fun

for ep in range(num_episodes):

    total_score = 0

    if ep == (num_episodes - 1):
        env = gym.wrappers.Monitor(env, '/tmp/cartpole-experiment-1', force=True)

    # Reset environment
    observation = env.reset()

    for tm in range(num_time_steps):

        # Choose between random or optimal control
        if random.random() < epsilon:
            # Get Random Action
            action = env.action_space.sample()

        else:
            # Get optimal action using current state

            # Flatten observation data
            obs_vector = np.expand_dims(observation, axis=0)

            # Get new value function at all control values
            val = [sess.run(vl_model,
                            feed_dict={vl_state: obs_vector,
                                       vl_action: np.expand_dims([a], axis=0)})
                   for a in range(env.action_space.n)]

            # Set next action to be the one maximizing value fun
            action = np.argmax(val)

        # Register last observation
        last_observation = observation

        # Advance Game with chosen action
        observation, reward, done, info = env.step(action)

        # Store Transition and Reward as tuple in array
        transition_array.append((last_observation, action, observation, reward, done))

        # Mini-batch Draw from Transition Array
        #   -- In the future merge this with next step and make more efficient
        #   -- Maybe use Numba and/or think of better structure
        mini_batch = random.choices(transition_array, k=num_mini_batch)
        mini_batch = np.asarray(mini_batch)

        # Format Mini Batch into np.arrays for TF update
        fmt_state, fmt_action, sims = batch_format(mini_batch, sess, vl_model, env.action_space.n)

        # Perform gradient ascent step
        sess.run(vl_optim, feed_dict={vl_state: fmt_state, vl_action: fmt_action, vl_sim: sims})

        # Break if over
        if done:
            break

    print(ep)




    # print(tm)
    # print(sess.run(vl_loss, feed_dict={vl_state: fmt_state, vl_action: fmt_action, vl_sim: sims}))
    # print(sess.run(vl_model, feed_dict={vl_state: fmt_state, vl_action: fmt_action}))
    # print(sims)
