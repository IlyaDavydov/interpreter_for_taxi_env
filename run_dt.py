import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import random

from features_extraction import extract_features

def run_dt(episodes, render=False, agent=None):

    env = gym.make('Taxi-v3', render_mode='human' if render else None)

    agent.load("taxi_decision_tree_model.pkl")

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]  # states: 
        terminated = False      # True when fall or reached goal
        truncated = False       # True when actions > 200

        rewards = 0
        steps = 0
        prev_action = None

        rewards_per_episode = np.zeros(episodes)

        while(not terminated and not truncated):
            
            action = agent.predict(state, env, prev_action)[0]

            new_state,reward,terminated,truncated,_ = env.step(action)

            # Если состояние не изменилось, делаем случайный шаг (но не зацикливаемся)
            max_attempts = 5  # Ограничение на количество попыток сменить состояние
            attempts = 0

            while new_state == state and attempts < max_attempts:
                action = random.randint(0, 3)  # Берем случайное действие
                new_state, reward, terminated, truncated, _ = env.step(action)
                attempts += 1   

            rewards += reward

            state = new_state

            prev_action = action

            steps += 1

            if steps > 25:
                truncated = True

        rewards_per_episode[i] = rewards

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig('taxi_test_dt.png')  
