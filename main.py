# importing the dependencies
import numpy as np  # to create the Q-table
import gym  # to create the Taxi Environment
import random  # to generate random numbers

# Creating the environment
env = gym.make("Taxi-v2")
env.render()

# Create the Q-table and initialize it
q_table = np.zeros((env.observation_space.n, env.action_space.n))
print("Initializing the Q table with zero values")
# print(q_table)  # Visualizing the Q-table

# Create the hyper-parameters which will be used to train our agent
TOTAL_EPISODES = 50000  # total episodes is the number of episodes used to train our algorithm
TEST_EPISODES = 100  # Total test episodes to test our algorithm
MAX_STEPS_PER_EPISODE = 99  # Max steps per episode

LEARNING_RATE = 0.7  # Learning rate
DISCOUNT_FACTOR = 0.618  # Discounting rate or gamma

# Exploration parameters
EPSILON = 1.0  # Exploration rate
MAX_EPSILON = 1.0  # Exploration probability at start
MIN_EPSILON = 0.01  # Minimum exploration probability
DECAY_RATE = 0.01  # Exponential decay rate for exploration

print("Starting to train the Agent!")
# implementing the Q-Algorithm
for episode in range(TOTAL_EPISODES):
    # Reset the environment
    state = env.reset()
    done = False

    for step in range(MAX_STEPS_PER_EPISODE):
        # First we randomize a number
        exp_exp_tradeoff = random.uniform(0, 1)

        # If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
        if exp_exp_tradeoff > EPSILON:
            action = np.argmax(q_table[state, :])

        # Else doing a random choice --> exploration
        else:
            action = env.action_space.sample()

        # Take the action (a) and observe the outcome state(s') and reward(r)
        new_state, reward, done, info = env.step(action)

        # Update Q(s,a) :- Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        q_table[state, action] = q_table[state, action] + LEARNING_RATE * (reward + DISCOUNT_FACTOR *
                                                                           np.max(q_table[new_state, :]) - q_table[
                                                                               state, action])
        # print(q_table)
        # Our new state is state
        state = new_state

        # If done: finish episode
        if done:
            break

    episode += 1

    epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-DECAY_RATE * episode)
print("Finished training the Agent")
print(q_table)

print(" Testing the agent on the Environment")

