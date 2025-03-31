"""
This script trains, tests, and runs a decision tree agent for the Taxi-v3 environment in OpenAI Gym.

Steps:
1. It trains a Q-agent on the Taxi-v3 environment for 15,000 episodes.
2. Generates data for the agent to train a decision tree.
3. Creates and trains a decision tree model (TaxiDecisionTree) on the generated data.
4. Saves the trained model as a pickle file for future use.
5. Runs the trained decision tree agent on the Taxi-v3 environment for a specified number of episodes and optionally renders the environment.
6. Prints the structure of the decision tree with actual feature names.

Instructions:
- **First-time execution:** If you are running the code for the first time, you don't need to change anything. 
The code will train the model from scratch, save it as a pickle file, and run the agent on the Taxi-v3 environment.

- **Subsequent executions:** read comments
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd

from trainQ import train_or_test_q
from data_creation import data_create
from features_extraction import extract_features
from taxi_decision_tree import TaxiDecisionTree
from run_dt import run_dt

env = gym.make('Taxi-v3')

# Comment out if you don't need to train the agent and already have a pickle file with the Q-table
train_or_test_q(episodes=15000, is_training=True, render=False)

# Uncomment if you want to see how the agent behaves when trained with Q-learning
#train_or_test_q(episodes=1, is_training=False, render=True) 

# Comment out if the Decision Tree agent is already trained and you have a pickle file with the decision tree
df = data_create(episodes=15000, render=False)

# It always remains uncommented
agent = TaxiDecisionTree(max_depth=16)

# Train model
# Comment out if the Decision Tree agent is already trained and you have a pickle file with the decision tree
agent.train(df)

# Save model
# Comment out if the Decision Tree agent is already trained and you have a pickle file with the decision tree
agent.save("taxi_decision_tree_model.pkl")

# Comment out if you don't want to observe the agent's behavior
run_dt(5, render=True, agent=agent)

# Uncomment if you want to see the agent's statistics over 15000 iterations (saved as png)
#run_dt(15000, render=False, agent=agent)

# Print the tree with actual feature names and actions
agent.print_tree(df=df)



