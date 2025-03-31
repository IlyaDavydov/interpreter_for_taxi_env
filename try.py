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

train_or_test_q(episodes=15000, is_training=True, render=False)
#train_or_test_q(episodes=1, is_training=False, render=True)

df = data_create(episodes=15000, render=False)

agent = TaxiDecisionTree(max_depth=64)

#train model
agent.train(df)

#save model
agent.save("taxi_decision_tree_model.pkl")

run_dt(5, render=True, agent=agent)



