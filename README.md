# Taxi Decision Tree Agent

This project implements a hybrid approach for solving the classic Taxi-v3 environment from OpenAI Gym using a combination of **Q-learning**, **interpretable feature engineering**, and **decision tree classification**. The goal is to train a Q-learning agent and then extract interpretable features to build a decision tree that mimics its behavior without relying on raw positional data.

## Overview

This project implements a similar algorithm from the paper **Interpretable and Editable Programmatic Tree Policies for Reinforcement Learning** ([link to paper](https://openreview.net/pdf?id=zafp5CwoTq)), with the distinction that the decision tree is partially clad, rather than fully (see the explanation in the **features** section).

The workflow is as follows:

1. Train a Q-learning agent on the Taxi-v3 environment (`trainQ.py`).
2. Generate training data using the Q-table and interpretable feature extraction (`data_creation.py`).
3. Train a decision tree model using the generated data (`taxi_decision_tree.py`).
4. Run and evaluate the decision tree agent in the environment (`run_dt.py` or `try.py`).
5. Visualize (`try.py`) and measure performance using reward plots.

## Project Setup Instructions

Follow the steps below to create a Conda environment and install the required dependencies for the project.

### 1. Create a New Conda Environment

First, create a new Conda environment with Python 3.10. Run the following commands:

```bash
conda create --name your_env_name python=3.10
```
pip install gymnasium numpy pandas matplotlib scikit-learn

### 2. Activate the Environment

After the environment is created, activate it by running:

```bash
conda activate your_env_name
```
This command switches to the environment you just created.

### 3. Install the Required Dependencies

To install the necessary dependencies, use the following command:

```bash
conda install gymnasium numpy pandas matplotlib scikit-learn
```

This will install the following Python packages:

- [ ] gymnasium: For interacting with the OpenAI Gym environments.
- [ ] numpy: For numerical computations and data manipulation.
- [ ] pandas: For data manipulation and analysis.
- [ ] matplotlib: For plotting and visualizations.
- [ ] scikit-learn: For machine learning tools, including the decision tree model.

### Running the Project

**To train from scratch:**

```bash
python try.py
```

We have provided detailed and clear instructions for running the project in the `try.py` file.
Therefore, simply review the file and run it in the way that best suits your needs and goals (whether you want to observe the agent's behavior or just test without playback, recording statistics; whether you already have pickle files with instructions, or if you are running the project from scratch for the first time, etc.).

## Project Structure

| File | Description |
|------|-------------|
| `trainQ.py` | Trains the taxi using Q-learning (100% accuracy, but not interpretable) |
| `data_creation.py` | Generates data based on Q-learning policy in test mode and extracts features |
| `features_extraction.py` | Helps extract interpretable features from the environment's state |
| `taxi_decision_tree.py` | Builds a decision tree based on the generated DataFrame |
| `run_dt.py` | Runs the simulation using the trained decision tree |
| `try.py` | Main script to run the full training and evaluation pipeline |

## Feature Engineering Philosophy

Instead of using raw positional data (e.g., taxi coordinates, passenger location), we focus on **interpretable features** such as:

- Whether the **distance to the passenger or destination** increases or decreases after taking an action.
- Whether a certain **movement is even possible** (e.g., is there a wall?).
- Whether the taxi is **at a special location**.
- Whether it’s time to **drop off the passenger**.
- Whether a previous action was performed.

This design helps keep the decision tree **interpretable and generalizable**, rather than overfitting to specific grid coordinates.

**Preferred**:  
_"Does the distance to the goal decrease when moving right? → Yes → Move right."_

**Avoided**:  
_"If position is (1,3) → Move right."_

Even though raw coordinates might lead to better accuracy, the resulting model is harder to interpret and tightly bound to the map structure.

## Evaluation

The decision tree model mimics the behavior of the Q-agent with a reasonable trade-off between accuracy and interpretability. 

## Model Persistence

- Q-table is saved to `taxi.pkl`
- Decision tree model is saved to `taxi_decision_tree_model.pkl`

## Notes

- The maximum depth of the decision tree is configurable.
- The system includes fallbacks in case the decision tree predicts invalid actions (e.g., retry with random choices).

## Future Applications

While the Taxi-v3 environment is a simplified, discrete grid-world, it offers a valuable abstraction for many real-world decision-making and planning problems—especially in domains involving **navigation, routing**, and **task execution** by autonomous agents.

The core task in Taxi-v3—picking up a passenger and dropping them off at a destination—closely mirrors real-life logistics scenarios, such as those faced by **autonomous taxis**, **ride-sharing services**, or **delivery drones**. By training agents in such environments and emphasizing **interpretable decision-making**, we can build systems that are both **effective and understandable**—crucial for deploying AI in **real-world, safety-critical contexts**.

One of the major contributions of this project is the **emphasis on interpretability**: instead of relying on opaque Q-tables or black-box neural networks, we extract **high-level, semantically meaningful** features.

These features represent **human-understandable** reasoning strategies, which are more transferable and easier to debug or audit. Avoiding raw state inputs (like absolute coordinates) ensures that the agent's behavior is not overfitted to one specific environment layout, increasing the generalizability of the learned policy.

Furthermore, this project could serve as a foundation for **integrating symbolic planners with language-based interfaces**. The inclusion of an interpreter module, capable of translating natural language commands into structured plans, opens the door for more intuitive and flexible human-agent interaction. For example, a future system could process commands such as “Pick up the passenger at the blue station and drop them at the green location” and automatically generate an executable, interpretable plan.

Such a setup has promising applications in:

- Human-robot collaboration: Enabling robots to follow natural-language instructions in warehouses, factories, or homes.

- Autonomous mobility systems: Providing safe, explainable decision-making for taxis or last-mile delivery bots.

- Education and research: Serving as a modular framework to study planning, RL, and explainability.

In essence, the Taxi-v3 environment acts as a sandbox for testing ideas that scale to the physical world, and by combining interpretable models with flexible instruction parsing, this project contributes to a growing effort in making AI not only powerful but also transparent and trustworthy.

## License

This project is for academic and research purposes.
