import gymnasium as gym
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

def extract_features(state, env):

    """
    Extracts features from the given state in the Taxi-v3 environment.

    Parameters:
    - state (int): Encoded state representation from the Taxi-v3 environment.
    - env (gym.Env): The Taxi-v3 environment instance.

    Description:
    - Decodes the state into taxi row/column, passenger location, and destination.
    - Computes Manhattan distances between the taxi, passenger, and destination.
    - Determines movement constraints based on the environment's layout.
    - Identifies whether the taxi is at a special location.
    - Checks if dropping off the passenger is the correct action.
    - Computes directionality features to indicate movement towards the passenger/destination.
    - Computes distance change for each possible action.
    - Returns a dictionary of extracted features.

    Returns:
    - dict: A dictionary containing extracted state features.
    """

    taxi_row, taxi_col, pass_loc, dest = env.unwrapped.decode(state)
    
     # Coordinates of pickup and drop-off locations
    locations = [(0, 0), (0, 4), (4, 0), (4, 3)]  # R, G, Y, B
    passenger_row, passenger_col = locations[pass_loc] if pass_loc < 4 else (taxi_row, taxi_col)
    dest_row, dest_col = locations[dest]

    # Manhattan distances
    taxi_to_passenger = abs(taxi_row - passenger_row) + abs(taxi_col - passenger_col)
    taxi_to_dest = abs(taxi_row - dest_row) + abs(taxi_col - dest_col)

    # Checking movement constraints for eastward direction
    can_move_east = taxi_col < 4 and not (
        (taxi_row == 3 and taxi_col == 0) or 
        (taxi_row == 4 and taxi_col == 0) or 
        (taxi_row == 3 and taxi_col == 2) or 
        (taxi_row == 4 and taxi_col == 2) or 
        (taxi_row == 0 and taxi_col == 1) or 
        (taxi_row == 1 and taxi_col == 1)
    )

    # Checking movement constraints for westward direction
    can_move_west = taxi_col > 0 and not (
        (taxi_row == 3 and taxi_col == 1) or 
        (taxi_row == 4 and taxi_col == 1) or 
        (taxi_row == 3 and taxi_col == 3) or 
        (taxi_row == 4 and taxi_col == 3) or 
        (taxi_row == 0 and taxi_col == 2) or 
        (taxi_row == 1 and taxi_col == 2)
    )

    # Checking boundaries for southward and northward movement
    can_move_south = taxi_row < 4
    can_move_north = taxi_row > 0


    # 1. Change in distance to the passenger's position if the passenger is not in the taxi
    def get_passenger_distance_change(action):
        d_row, d_col = action
        new_row, new_col = taxi_row + d_row, taxi_col + d_col
        if 0 <= new_row <= 4 and 0 <= new_col <= 4:
            new_taxi_to_passenger = abs(new_row - passenger_row) + abs(new_col - passenger_col)
            return new_taxi_to_passenger - taxi_to_passenger
        return 0 # If movement is out of bounds, return 0 (no change)

    # 2. Change in distance to the destination if the passenger is already in the taxi
    def get_dest_distance_change(action):
        d_row, d_col = action
        new_row, new_col = taxi_row + d_row, taxi_col + d_col
        if 0 <= new_row <= 4 and 0 <= new_col <= 4:
            new_taxi_to_dest = abs(new_row - dest_row) + abs(new_col - dest_col)
            return new_taxi_to_dest - taxi_to_dest
        return 0  # If movement is out of bounds, return 0 (no change)
    
     # 3. Is the passenger in the taxi?
    passenger_in_taxi = int(pass_loc == 4)

    # 4. Is the taxi at one of the special locations?
    taxi_at_special_location = int((taxi_row, taxi_col) in locations)

    # 5. Can the action be performed?
    action_is_possible_south = int(can_move_east)
    action_is_possible_north = int(can_move_north)
    action_is_possible_west = int(can_move_west)
    action_is_possible_east = int(can_move_east)

     # 6. Direction from the taxi to the passenger, if the passenger is not in the taxi
    direction_to_passenger = -1
    if pass_loc != 4:
        if passenger_row < taxi_row and passenger_col < taxi_col:
            direction_to_passenger = 1  # up-left
        elif passenger_row < taxi_row and passenger_col > taxi_col:
            direction_to_passenger = 2  # up-right
        elif passenger_row > taxi_row and passenger_col < taxi_col:
            direction_to_passenger = 0  # down-left
        elif passenger_row > taxi_row and passenger_col > taxi_col:
            direction_to_passenger = 3  # down-right
        elif passenger_row == taxi_row and passenger_col < taxi_col:
            direction_to_passenger = 4  # left
        elif passenger_row == taxi_row and passenger_col > taxi_col:
            direction_to_passenger = 5  # right
        elif passenger_col == taxi_col and passenger_row < taxi_row:
            direction_to_passenger = 6  # up
        elif passenger_col == taxi_col and passenger_row > taxi_row:
            direction_to_passenger = 7  # down

    # 7. Direction from the taxi to the destination, if the passenger is in the taxi
    direction_to_dest = -1
    if passenger_in_taxi:
        if dest_row < taxi_row and dest_col < taxi_col:
            direction_to_dest = 1  # up-left
        elif dest_row < taxi_row and dest_col > taxi_col:
            direction_to_dest = 2  # up-right
        elif dest_row > taxi_row and dest_col < taxi_col:
            direction_to_dest = 0  # down-left
        elif dest_row > taxi_row and dest_col > taxi_col:
            direction_to_dest = 3  # down-right
        elif dest_row == taxi_row and dest_col < taxi_col:
            direction_to_dest = 4  # left
        elif dest_row == taxi_row and dest_col > taxi_col:
            direction_to_dest = 5  # right
        elif dest_col == taxi_col and dest_row < taxi_row:
            direction_to_dest = 6  # up
        elif dest_col == taxi_col and dest_row > taxi_row:
            direction_to_dest = 7  # down

    # 8. Previous action (will be updated depending on the state)
    previous_action = None  # will update depend on last state

    # 9. # Should the passenger be dropped off?
    should_drop_passenger = passenger_in_taxi and (taxi_row == dest_row and taxi_col == dest_col)


    # Compute distance changes for each possible action
    actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # south, north, east, west
    distance_changes_passenger = [get_passenger_distance_change(action) for action in actions]
    distance_changes_dest = [get_dest_distance_change(action) for action in actions]

    # Is the taxi at position (2,1)?
    # In this position, the agent often got stuck in a loop. We introduced being in this position as a separate feature,
    # which significantly improved the model. Without this feature, it still worked quite well, 
    # but with it, the performance was noticeably better. 
    # Of course, this is somewhat of a "cheat" and not a truly universal feature, as it may vary depending on the map,
    # which is not ideal. However, we could generalize this to a real-world problem 
    # by testing different maps and passing such potential loop-prone positions as parameters. 
    # This would make it a more natural feature.

    taxi_in_2_1 = taxi_row == 2 and taxi_col == 1

     # Construct the final feature dictionary
    features = {
        "passenger_in_taxi": passenger_in_taxi,
        "taxi_at_special_location": taxi_at_special_location,
        "action_is_possible_south": action_is_possible_south,
        "action_is_possible_north": action_is_possible_north,
        "action_is_possible_west": action_is_possible_west,
        "action_is_possible_east": action_is_possible_east,
        "should_drop_passanger": should_drop_passenger,
        "taxi_in_2_1": taxi_in_2_1, 
        "previous_action": previous_action,
    }

    # Add distance changes for each action
    features.update({
        "distance_change_passenger_south": distance_changes_passenger[0],
        "distance_change_passenger_north": distance_changes_passenger[1],
        "distance_change_passenger_east": distance_changes_passenger[2],
        "distance_change_passenger_west": distance_changes_passenger[3],
        "distance_change_dest_south": distance_changes_dest[0],
        "distance_change_dest_north": distance_changes_dest[1],
        "distance_change_dest_east": distance_changes_dest[2],
        "distance_change_dest_west": distance_changes_dest[3],
    })

    return features

