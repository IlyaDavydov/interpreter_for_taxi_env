import gymnasium as gym
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

def extract_features(state, env):
    taxi_row, taxi_col, pass_loc, dest = env.unwrapped.decode(state)
    
    # Координаты посадки и высадки
    locations = [(0, 0), (0, 4), (4, 0), (4, 3)]  # R, G, Y, B
    passenger_row, passenger_col = locations[pass_loc] if pass_loc < 4 else (taxi_row, taxi_col)
    dest_row, dest_col = locations[dest]

    # Манхэттенские расстояния
    taxi_to_passenger = abs(taxi_row - passenger_row) + abs(taxi_col - passenger_col)
    taxi_to_dest = abs(taxi_row - dest_row) + abs(taxi_col - dest_col)

    # Проверка для движения на восток (восточные ограничения)
    can_move_east = taxi_col < 4 and not (
        (taxi_row == 3 and taxi_col == 0) or 
        (taxi_row == 4 and taxi_col == 0) or 
        (taxi_row == 3 and taxi_col == 2) or 
        (taxi_row == 4 and taxi_col == 2) or 
        (taxi_row == 0 and taxi_col == 1) or 
        (taxi_row == 1 and taxi_col == 1)
    )

    # Проверка для движения на запад (западные ограничения)
    can_move_west = taxi_col > 0 and not (
        (taxi_row == 3 and taxi_col == 1) or 
        (taxi_row == 4 and taxi_col == 1) or 
        (taxi_row == 3 and taxi_col == 3) or 
        (taxi_row == 4 and taxi_col == 3) or 
        (taxi_row == 0 and taxi_col == 2) or 
        (taxi_row == 1 and taxi_col == 2)
    )

    # Проверка границ для движения на юг и на север
    can_move_south = taxi_row < 4
    can_move_north = taxi_row > 0


    # 1. Изменение расстояния до позиции пассажира, если пассажир не в такси
    def get_passenger_distance_change(action):
        d_row, d_col = action
        new_row, new_col = taxi_row + d_row, taxi_col + d_col
        if 0 <= new_row <= 4 and 0 <= new_col <= 4:
            new_taxi_to_passenger = abs(new_row - passenger_row) + abs(new_col - passenger_col)
            return new_taxi_to_passenger - taxi_to_passenger
        return 0  # Если движение выходит за границы, возвращаем 0 (не изменяется)

    # 2. Изменение расстояния до финиша, если пассажир уже в такси
    def get_dest_distance_change(action):
        d_row, d_col = action
        new_row, new_col = taxi_row + d_row, taxi_col + d_col
        if 0 <= new_row <= 4 and 0 <= new_col <= 4:
            new_taxi_to_dest = abs(new_row - dest_row) + abs(new_col - dest_col)
            return new_taxi_to_dest - taxi_to_dest
        return 0  # Если движение выходит за границы, возвращаем 0 (не изменяется)

    # 3. Пассажир в такси?
    passenger_in_taxi = int(pass_loc == 4)

    # 4. Находится ли такси на одной из специальных локаций
    taxi_at_special_location = int((taxi_row, taxi_col) in locations)

    # 5. Можно ли провести действие
    action_is_possible_south = int(can_move_east)
    action_is_possible_north = int(can_move_north)
    action_is_possible_west = int(can_move_west)
    action_is_possible_east = int(can_move_east)

    # 6. Направление от такси до пассажира, если пассажир не в такси
    direction_to_passenger = -1
    if pass_loc != 4:
        if passenger_row < taxi_row and passenger_col < taxi_col:
            direction_to_passenger = 1  # вверх-влево
        elif passenger_row < taxi_row and passenger_col > taxi_col:
            direction_to_passenger = 2  # вверх-вправо
        elif passenger_row > taxi_row and passenger_col < taxi_col:
            direction_to_passenger = 0  # вниз-влево
        elif passenger_row > taxi_row and passenger_col > taxi_col:
            direction_to_passenger = 3  # вниз-вправо
        elif passenger_row == taxi_row and passenger_col < taxi_col:
            direction_to_passenger = 4  # влево
        elif passenger_row == taxi_row and passenger_col > taxi_col:
            direction_to_passenger = 5  # вправо
        elif passenger_col == taxi_col and passenger_row < taxi_row:
            direction_to_passenger = 6  # вверх
        elif passenger_col == taxi_col and passenger_row > taxi_row:
            direction_to_passenger = 7  # вниз

    # 7. Направление от такси до финиша, если пассажир в такси
    direction_to_dest = -1
    if passenger_in_taxi:
        if dest_row < taxi_row and dest_col < taxi_col:
            direction_to_dest = 1  # вверх-влево
        elif dest_row < taxi_row and dest_col > taxi_col:
            direction_to_dest = 2  # вверх-вправо
        elif dest_row > taxi_row and dest_col < taxi_col:
            direction_to_dest = 0  # вниз-влево
        elif dest_row > taxi_row and dest_col > taxi_col:
            direction_to_dest = 3  # вниз-вправо
        elif dest_row == taxi_row and dest_col < taxi_col:
            direction_to_dest = 4  # влево
        elif dest_row == taxi_row and dest_col > taxi_col:
            direction_to_dest = 5  # вправо
        elif dest_col == taxi_col and dest_row < taxi_row:
            direction_to_dest = 6  # вверх
        elif dest_col == taxi_col and dest_row > taxi_row:
            direction_to_dest = 7  # вниз

    # 8. Предыдущее действие
    previous_action = None  # Будет обновляться в зависимости от состояния

    should_drop_passenger = passenger_in_taxi and (taxi_row == dest_row and taxi_col == dest_col)


    # Для каждого возможного действия считаем изменение расстояния
    actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # south, north, east, west
    distance_changes_passenger = [get_passenger_distance_change(action) for action in actions]
    distance_changes_dest = [get_dest_distance_change(action) for action in actions]

    taxi_in_2_1 = taxi_row == 2 and taxi_col == 1

    # Формируем финальные признаки
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

    # Добавляем изменения в расстояния для каждого действия
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


