# Taxi Decision Tree Agent

This project implements a hybrid approach for solving the classic Taxi-v3 environment (https://www.gymlibrary.dev/environments/toy_text/taxi/) from OpenAI Gym using a combination of **Q-learning**, **interpretable feature engineering**, and **decision tree classification**. The goal is to train a Q-learning agent and then extract interpretable features to build a decision tree that mimics its behavior without relying on raw positional data.

## Overview

This project implements a similar algorithm from the paper **Interpretable and Editable Programmatic Tree Policies for Reinforcement Learning** ([link to paper](https://openreview.net/pdf?id=zafp5CwoTq)), with the distinction that the decision tree is partially oblique, rather than fully (see the explanation in the **features** section).

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

**For more detailed information and specific implementation, refer to the file `features_extraction.py`**

The accuracy and quality of the decision tree depend on the features we choose. Therefore, **this section is critically important** – if you want to improve the model, you need to understand which features most accurately affect the outcome and either come up with them or remove unnecessary ones.

We want to mention upfront that the task of building a decision tree for a specific map can be easily solved by using features such as the position of the taxi, the passenger, and the goal. This would create an overfitted decision tree that works perfectly on this map. However, its issues would be twofold: firstly, it would not be universal, and secondly, it would be harder to interpret (we would need to use absolute features, such as "if the position is (3,3), do action 'right'") instead of relative features like "will the distance to the goal decrease if I do action 'right'?" which is independent of the specific map and easier to understand for a human. 

Our main goal is not to achieve exact results but to maintain good interpretability while keeping similar results.

Moreover, this effectively makes the tree partially *oblique*, without requiring us to implement a complex ObliqueTree model. Even though the feature is formally one, it inherently includes several different features — for example, the current position, the future position, and the possible future action (essentially, it calculates the difference in Manhattan distances between the current and future possible states for each action).

### Features Used in the Decision Tree Model

- **Passenger Distance Change**: 
  - Measures how the distance to the passenger changes after taking an action.

- **Destination Distance Change**: 
  - Measures how the distance to the destination changes after taking an action, but only when the passenger is in the taxi.

- **Movement Feasibility**: 
  - Checks whether certain movements (south, north, east, west) are possible based on the taxi’s current position and the environment's layout.

- **Special Locations**: 
  - Indicates whether the taxi is at a special location, such as one of the predefined pickup/drop-off points.

- **Drop-off Condition**: 
  - Identifies whether it is time to drop off the passenger, based on the taxi's position and whether the passenger is in the vehicle.

- **Previous Action**: 
  - Tracks the last action performed by the taxi (useful for creating a sequence-based model).

- **Directionality**: 
  - Determines the direction the taxi should move in relation to the passenger or destination, based on their relative positions.

- **Taxi in Specific Position (2,1)**: 
  - Checks if the taxi is in a known position where the agent often gets stuck, which helps avoid loops in the learning process.


This design helps keep the decision tree **interpretable and generalizable**, rather than overfitting to specific grid coordinates.

**Preferred**:  
_"Does the distance to the goal decrease when moving right? → Yes → Move right."_

**Avoided**:  
_"If position is (1,3) → Move right."_

Even though raw coordinates might lead to better accuracy, the resulting model is harder to interpret and tightly bound to the map structure.

## Results & Demonstration

Overall, the results are positive, and the approach makes sense. However, some unresolved issues still need to be addressed in the future. One recurring problem is that the agent sometimes gets stuck in a loop. This likely happens because, in certain positions, the optimal move requires temporarily moving away from the goal to bypass an obstacle. Instead, the agent keeps moving along the obstacle, initially reducing the distance to the goal but ultimately leading to an error. Additionally, the agent occasionally follows suboptimal paths.

Despite these issues, we achieved the following goals:
1. The agent almost always reaches the goal one way or another.
2. In most cases, the agent selects an optimal path.
3. Considering points 1 and 2, the agent remains interpretable and generalizable.

### Agent Behavior
Here you can see the agent's behavior:  
**_INSERT VIDEO HERE_**

### Performance Comparison

We developed a metric to compare Q-Learning and Decision Tree approaches: the average number of steps taken over 100 iterations. The assumption is that an optimally performing agent will take the least number of steps on average. If the agent gets stuck in a loop or fails to find the goal, the number of steps will be significantly higher.

### Results Comparison
**Q-Learning vs. Decision Tree**  
_Photo 1_ | _Photo 2_  
**_INSERT IMAGES HERE_**

As seen in the results, the performance is generally good. The Q-Learning agent, after 10,000 iterations and achieving 100% accuracy on this map, reaches the goal in an average of 12 steps. Meanwhile, the Decision Tree Agent typically reaches the goal in 18-19 steps (with a fluctuating graph due to occasional looping). This is a promising result, as the difference is not too large, and the Decision Tree Agent remains generalizable, interpretable, and potentially improvable. We believe that the looping issue can be resolved in future iterations.

### Decision Tree Visualization
The Decision Tree class includes a `print_tree` function, allowing users to inspect how the agent makes decisions. Below is a sample output:

_Photo 3_  
**_INSERT IMAGE HERE_**
 

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
