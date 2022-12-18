import gym
import json
import numpy as np

# create the Gym environment
env = gym.make('HalfCheetah-v4')

# set the number of steps to collect data for
num_steps = 1000

# initialize a list to hold the data
data = []

# run the environment for the specified number of steps
for i in range(num_steps):
    # reset the environment to its initial state
    state = env.reset()

    # collect data from the environment
    data.append(state)

with open('test.npy', 'wb') as f:
    np.save(f, data)
# save the data to a file
with open('offline_data.json', 'w') as f:
    json.dump(data, f)