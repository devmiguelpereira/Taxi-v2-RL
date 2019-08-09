# importing the dependencies
import numpy as np  # to create the Q-table
import gym          # to create the Taxi Environment
import random       # to generate random numbers

# Creating the environment
env = gym.make("Taxi-v2")
env.render()

# Create the Q-table and initialize it
q_table = np.zeros((env.observation_space.n, env.action_space.n))
print("Initializing the Q table with zero values")
# print(q_table)  # Visualizing the Q-table

# Create the hyper-parameters which will be used to train our agent
EPISODES = 50000     # total episodes is the number of episodes used to train our algorithm
TEST_EPISODES = 100  # Total test episodes to test our algorithm
MAX_STEPS_PER_EPISODE = 99  # Max steps per episode

LEARNING_RATE = 0.7  # Learning rate
DISCOUNT_FACTOR = 0.618  # Discounting rate or gamma

# Exploration parameters
EPSILON = 1.0       # Exploration rate
MAX_EPSILON = 1.0   # Exploration probability at start
MIN_EPSILON = 0.01  # Minimum exploration probability
DECAY_RATE = 0.01   # Exponential decay rate for exploration

# implementing the Q-Algorithm
