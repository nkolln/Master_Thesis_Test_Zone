# import gym

# env = gym.make("HalfCheetah-v4", render_mode="human")
# print(env)

# print('hello world')
# print()

import gym
import d4rl
print('Starting')
env = gym.make("halfcheetah-medium-v2")


env.action_space.seed(42)
env=env.unwrapped
dataset = env.get_dataset()
print(dataset)

