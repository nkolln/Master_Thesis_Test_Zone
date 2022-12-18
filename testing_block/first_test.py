# import gym

# env = gym.make("HalfCheetah-v4", render_mode="human")
# print(env)

# print('hello world')
# print()
import pandas as pd

import gym
import d4rl
import collections
import pickle
import numpy as np

print('Starting')
env_name = 'halfcheetah'
dataset_type = 'expert'
name = f'{env_name}-{dataset_type}-v2'
env = gym.make("halfcheetah-expert-v2")
dataset = env.get_dataset()


N = dataset['rewards'].shape[0]
data_ = collections.defaultdict(list)
print(data_)

use_timeouts = False
if 'timeouts' in dataset:
    use_timeouts = True

episode_step = 0
paths = []
for i in range(N):
    done_bool = bool(dataset['terminals'][i])
    if use_timeouts:
        final_timestep = dataset['timeouts'][i]
    else:
        final_timestep = (episode_step == 1000-1)
    for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
        data_[k].append(dataset[k][i])
    if done_bool or final_timestep:
        episode_step = 0
        episode_data = {}
        for k in data_:
            episode_data[k] = np.array(data_[k])
        paths.append(episode_data)
        data_ = collections.defaultdict(list)
    episode_step += 1

returns = np.array([np.sum(p['rewards']) for p in paths])
num_samples = np.sum([p['rewards'].shape[0] for p in paths])
print(f'Number of samples collected: {num_samples}')
print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')
with open('test.npy', 'wb') as f:
    np.save(f, paths)
# df = pd.DataFrame(paths)
# df.to_csv('df_out.csv')
# with open(f'{name}.pkl', 'wb') as f:
#     pickle.dump(paths, f)
