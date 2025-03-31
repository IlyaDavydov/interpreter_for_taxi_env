import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd

from features_extraction import extract_features

def data_create(episodes, render=False):

    env = gym.make('Taxi-v3', render_mode='human' if render else None)

    data = []

    f = open('taxi.pkl', 'rb')
    q = pickle.load(f)
    f.close()

    learning_rate_a = 0.9 # alpha or learning rate
    discount_factor_g = 0.9 # gamma or discount rate. Near 0: more weight/reward placed on immediate state. Near 1: more on future state.
    epsilon = 1         # 1 = 100% random actions
    epsilon_decay_rate = 0.0001        # epsilon decay rate. 1/0.0001 = 10,000
    rng = np.random.default_rng()   # random number generator

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]  # states: 
        terminated = False      # True when fall or reached goal
        truncated = False       # True when actions > 200

        episode_data = []
        steps = 0
        prev_action = None

        rewards = 0
        while(not terminated and not truncated):

            action = np.argmax(q[state,:])
            features = extract_features(state, env)
            features["previous_action"] = prev_action
            features["action"] = action

            new_state, reward, terminated, truncated,_ = env.step(action)

            rewards += reward

            state = new_state

            episode_data.append(features)

            prev_action = action

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        data.extend(episode_data)

        if(epsilon==0):
            learning_rate_a = 0.0001


        rewards_per_episode[i] = rewards

    env.close()

    df = pd.DataFrame(data)
    return df  
