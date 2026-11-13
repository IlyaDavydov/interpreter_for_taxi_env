import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import random

from features_extraction import extract_features

def run_dt(episodes, render=False, agent=None):
    """
    Runs the Taxi-v3 environment using a Decision Tree agent.
    
    Parameters:
    - episodes (int): Number of episodes to run.
    - render (bool): Whether to render the environment visually.
    - agent: Pre-trained decision tree agent.
    
    Saves a plot of the average number of steps per 100 episodes.
    """

    env = gym.make('Taxi-v3', render_mode='human' if render else None)

    agent.load("taxi_decision_tree_model.pkl")

    rewards_per_episode = np.zeros(episodes)

    steps_per_episode = np.zeros(episodes)  # Stores the number of steps per episode

    for i in range(episodes):
        state = env.reset()[0]  # states: 
        terminated = False      # True when fall or reached goal
        truncated = False       # True when actions > 200

        rewards = 0
        steps = 0
        prev_action = None

        while(not terminated and not truncated):
            
            action = agent.predict(state, env, prev_action)[0]

            new_state, reward, terminated, truncated, _ = env.step(action)
            rewards += reward
            steps += 1

            max_attempts = 5
            attempts = 0

            while (new_state == state and not terminated and not truncated and attempts < max_attempts):
                action = random.randint(0, 3)
                new_state, reward, terminated, truncated, _ = env.step(action)
                rewards += reward
                steps += 1
                attempts += 1

            state = new_state
            prev_action = action
            if render and steps > 25:
                truncated = True

        rewards_per_episode[i] = rewards
        steps_per_episode[i] = steps  # Store the number of steps   

    env.close()

    sum_rewards = np.zeros(episodes)
    avg_steps = np.zeros(episodes)
    for t in range(episodes):
        avg_steps[t] = np.mean(steps_per_episode[max(0, t - 100):(t + 1)])

    plt.plot(avg_steps)
    plt.xlabel("Episodes")
    plt.ylabel("Avg Steps per 100 Episodes")
    plt.title("Decision Tree Results")
    plt.savefig('taxi_dt_steps.png')  

