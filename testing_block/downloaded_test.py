import gym
env = gym.make("HalfCheetah-v4", render_mode='human')


env.action_space.seed(42)
observation, info = env.reset(seed=42)
print(observation)
print(info)