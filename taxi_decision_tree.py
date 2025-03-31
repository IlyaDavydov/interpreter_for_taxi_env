import pandas as pd
import numpy as np
import pickle

from sklearn.tree import DecisionTreeClassifier

from features_extraction import extract_features

class TaxiDecisionTree:
    """
    A decision tree-based agent for solving the Taxi-v3 environment.
    This class trains a decision tree model to predict optimal actions based on extracted state features.
    """
        
    def __init__(self, max_depth=8, random_state=42):
        """
        Initializes the decision tree model with given parameters.
        :param max_depth: The maximum depth of the decision tree.
        :param random_state: Seed for random number generator (ensures reproducibility).
        """
        self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)

    def train(self, df):
        """
        Trains the decision tree model using a provided DataFrame.
        :param df: pandas DataFrame containing extracted features and corresponding actions.
        """
        X = df.iloc[:, :-1].values  # Extract all columns except the last one (features)
        y = df.iloc[:, -1].values   # Last column contains the action labels

        # Train the decision tree model
        self.model.fit(X, y)

    def predict(self, state, env, prev_action):
        """
        Predicts the best action given the current state.
        :param state: The current state of the environment.
        :param env: The Gym environment instance.
        :param prev_action: The previous action taken (used as an additional feature).
        :return: The predicted action.
        """
        features_data = extract_features(state, env) # Extract features from the state
        features_data["previous_action"] = prev_action # Include previous action as a feature 
        features = pd.DataFrame([features_data], index=[0])
        return self.model.predict(features)
    
    def print_tree(self, node=0, depth=0, df=None):
        """
        Recursively prints the structure of the decision tree with actual feature names.
        :param node: The current node index (starts at root, 0).
        :param depth: Current depth level for indentation.
        :param df: DataFrame containing feature names (optional, used for clarity).
        """

        tree = self.model.tree_

        indent = "  " * depth  # Indentation for visualizing tree levels

        feature_names = df.columns  # Get feature names from DataFrame
        actions = {
            0: "move south",
            1: "move north",
            2: "move east",
            3: "move west",
            4: "pickup passenger",
            5: "drop off passenger"
        }

        if tree.feature[node] != -2:  # Not a leaf node
            feature_name = feature_names[tree.feature[node]]  # Use the actual feature name
            print(f"{indent}Node {node}: checking {feature_name} <= {tree.threshold[node]:.3f}")
            self.print_tree(node=tree.children_left[node], depth=depth + 1, df=df)  # Left child
            self.print_tree(node=tree.children_right[node], depth=depth + 1, df=df)  # Right child
        else:  # Leaf node
            action = actions.get(np.argmax(tree.value[node]), f"unknown action {np.argmax(tree.value[node])}")
            print(f"{indent}Leaf {node}: action {action}")




    def save(self, filename):
        """
        Saves the trained model to a file.
        :param filename: The name of the file to save the model.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, filename):
        """
        Loads a trained model from a file.
        :param filename: The name of the file to load the model from.
        """
        with open(filename, 'rb') as f:
            self.model = pickle.load(f)

