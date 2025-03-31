import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def train_or_test_q(episodes, is_training=True, render=False):
    """
    Trains or tests a Q-learning agent on the Taxi-v3 environment.

    Parameters:
    - episodes (int): Number of episodes to run.
    - is_training (bool, optional): If True, the function trains the agent; otherwise, it tests a pre-trained model. Default is True.
    - render (bool, optional): If True, the environment is rendered during execution. Default is False.

    Description:
    - If `is_training` is True, initializes a Q-table with zeros and updates it using the Q-learning algorithm.
    - If `is_training` is False, loads a pre-trained Q-table from 'taxi.pkl'.
    - Implements an epsilon-greedy strategy for action selection.
    - Decays epsilon over time to shift from exploration to exploitation.
    - Tracks and plots the average number of steps per 100 episodes.
    - Saves the trained Q-table to 'taxi.pkl' after training.
    - Saves a graph showing training/testing progress.
    """
        
    env = gym.make('Taxi-v3', render_mode='human' if render else None)

    if(is_training):
        q = np.zeros((env.observation_space.n, env.action_space.n)) # init a 500 x 6 array
    else:
        f = open('taxi.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.9 # alpha or learning rate
    discount_factor_g = 0.9 # gamma or discount rate. Near 0: more weight/reward placed on immediate state. Near 1: more on future state.
    epsilon = 1         # 1 = 100% random actions
    epsilon_decay_rate = 0.0001        # epsilon decay rate. 1/0.0001 = 10,000
    rng = np.random.default_rng()   # random number generator

    rewards_per_episode = np.zeros(episodes)
    steps_per_episode = np.zeros(episodes)  # Храним количество шагов

    for i in range(episodes):
        state = env.reset()[0]  # states: 
        terminated = False      # True when fall or reached goal
        truncated = False       # True when actions > 200

        rewards = 0
        steps = 0

        while(not terminated and not truncated):

            if is_training and rng.random() < epsilon:
                action = env.action_space.sample() # actions
            else:
                action = np.argmax(q[state,:])

            new_state,reward,terminated,truncated,_ = env.step(action)

            rewards += reward

            if is_training:
                q[state,action] = q[state,action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action]
                )

            state = new_state

            steps += 1

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if(epsilon==0):
            learning_rate_a = 0.0001

        steps_per_episode[i] = steps  # save the count steps    


        rewards_per_episode[i] = rewards

    env.close()

    sum_rewards = np.zeros(episodes)
    avg_steps = np.zeros(episodes)
    for t in range(episodes):
        avg_steps[t] = np.mean(steps_per_episode[max(0, t - 100):(t + 1)])

    plt.plot(avg_steps)
    plt.xlabel("Episodes")
    plt.ylabel("Avg Steps per 100 Episodes")
    plt.title("Learning Progress")

    if is_training:
        plt.savefig('taxi_train_steps.png')
    else:
        plt.savefig('taxi_test_steps.png') 

    if is_training:
        f = open("taxi.pkl","wb")
        pickle.dump(q, f)
        f.close()

