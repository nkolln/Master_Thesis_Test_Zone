import gym

# create the environment
env = gym.make('halfcheetah')

# create a buffer to hold the experiences
buffer = []

# create a random seed
random_seed = 12345

# set the random seed
env.seed(random_seed)
env.
# generate the data
for _ in range(1000):
    # reset the environment
    state = env.reset()

    # generate experiences until the episode is done
    while True:
        # choose an expert action
        action = env.get_expert_action(state)

        # take the action and observe the next state and reward
        next_state, reward, done, _ = env.step(action)

        # add the experience to the buffer
        buffer.append((state, action, reward, next_state, done))

        # update the state
        state = next_state

        # if the episode is done, break the loop
        if done:
            break

# save the buffer to a file
#np.save('data.npy', buffer)
