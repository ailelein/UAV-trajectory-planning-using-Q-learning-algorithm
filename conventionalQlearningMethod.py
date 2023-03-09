import math
import numpy as np
import csv
import classical_hexegon_check
from sklearn import linear_model
import pandas as pd

HexagonCoordinate = classical_hexegon_check.hexagon_coordinates
HexagonHeight = classical_hexegon_check.hexagon_heights
HexagonSensor = classical_hexegon_check.Sensors_in_hexagon


def number_of_sensor_function():
    sensor_num_max = 500
    sensor_num_min = 2
    for i in range(HexagonNumber):
        for j in range(2):
            if HexagonSensor[i][j] == 0:
                sensor_per_cell[i][j] = 0
            elif 0 < HexagonSensor[i][j] <= sensor_num_max:
                sensor_per_cell[i][j] = (((sensor_num_max - sensor_num_min) * HexagonSensor[i][j]) + (
                        sensor_num_max * (sensor_num_min - 1))) / (sensor_num_max - 1)
            elif HexagonSensor[i][j] > sensor_num_max:
                sensor_per_cell[i][j] = sensor_num_max


def sensing_reward(HexagonID, UAVCurrentTime):
    if (prev_covered_time[HexagonID] == 0) or ((UAVCurrentTime - prev_covered_time[HexagonID]) >= Recovery_time1):
        SenValType1 = sen_val_max1
    elif (UAVCurrentTime - prev_covered_time[HexagonID]) < Recovery_time1 / 2:
        SenValType1 = sen_val_min1
    elif (Recovery_time1 / 2) <= (UAVCurrentTime - prev_covered_time[HexagonID]) < Recovery_time1:
        SenValType1 = (sen_val_max1 - sen_val_min1) / 2

    if (prev_covered_time[HexagonID] == 0) or ((UAVCurrentTime - prev_covered_time[HexagonID]) >= Recovery_time2):
        SenValType2 = sen_val_max2
    elif (UAVCurrentTime - prev_covered_time[HexagonID]) < Recovery_time2 / 2:
        SenValType2 = sen_val_min2

    elif (Recovery_time2 / 2) <= (UAVCurrentTime - prev_covered_time[HexagonID]) < Recovery_time2:
        SenValType2 = (sen_val_max2 - sen_val_min2) / 2

    sensing_reward_value = ((HexagonSensor[HexagonID][0] * SenValType1 * sensor_per_cell[HexagonID][0]) + (
                HexagonSensor[HexagonID][1] * SenValType2 * sensor_per_cell[HexagonID][1])) / SensingNor
    return sensing_reward_value


def reward(HexagonID, NewHexagonID, UAVCurrentTime):
    sensing_reward_value = sensing_reward(NewHexagonID, UAVCurrentTime)
    en_reward_value = energy_reward(HexagonID, NewHexagonID)
    reward_value = (weight_s * sensing_reward_value) - (weight_e * en_reward_value)
    return reward_value


def energy_reward(HexagonID, NewHexagonID):
    energy_reward_value = energy_fly(HexagonID, NewHexagonID) / EnergyNor
    return energy_reward_value


def energy_fly(HexagonID, NewHexagonID):
    h1 = HexagonHeight[HexagonID]
    h2 = HexagonHeight[NewHexagonID]
    moving_time_value = moving_time(HexagonID, NewHexagonID)
    if h2 > h1:
        energy_fly_value = (315 * (h2 - h1) - 211.261) + (
                308.709 * moving_time_value - 0.852)
    elif h2 < h1:
        energy_fly_value = (68.956 * (h1 - h2) - 65.183) + (
                308.709 * moving_time_value - 0.852)
    elif h1 == h2:
        energy_fly_value = 308.709 * moving_time_value - 0.852
    return energy_fly_value


def staying_time(HexagonID):
    sensor_list = HexagonSensor[HexagonID]
    staying_time_value = (sensor_list[0] + sensor_list[1]) * av_pc_dlr_time
    return staying_time_value


def PredictedHexagonSensor(HexagonID, SensorType):
    Z_Prediction = []
    X_Prediction = []
    Y_Prediction = []
    y_max = 973.7
    x_max = 951.5

    NeighbourCoordinates = []
    coor_one = HexagonCoordinate[HexagonID]

    if coor_one[1] + shortdiameter + 1 < y_max:  # ?
        neig_y = coor_one[1] + shortdiameter
        neig_x = coor_one[0]
        NeighbourCoordinates.append([neig_x, neig_y])
    if (coor_one[1] + (shortdiameter / 2) + 1 < y_max) and (
            (coor_one[0] + longdiameter / 2 + longdiameter / 4) + 1 < x_max):
        neig_y = coor_one[1] + (shortdiameter / 2)
        neig_x = coor_one[0] + longdiameter / 2 + longdiameter / 4
        NeighbourCoordinates.append([neig_x, neig_y])
    if (coor_one[1] - shortdiameter / 2 - 1 > 0) and ((coor_one[0] + longdiameter / 2 + longdiameter / 4) + 1 < x_max):
        neig_y = coor_one[1] - shortdiameter / 2
        neig_x = coor_one[0] + longdiameter / 2 + longdiameter / 4
        NeighbourCoordinates.append([neig_x, neig_y])
    if (coor_one[1] - shortdiameter - 1 > 0):
        neig_y = coor_one[1] - shortdiameter
        neig_x = coor_one[0]
        NeighbourCoordinates.append([neig_x, neig_y])
    if (coor_one[1] - (shortdiameter / 2) - 1 > 0) and ((coor_one[0] - longdiameter / 2 - longdiameter / 4) - 1 > 0):
        neig_y = coor_one[1] - (shortdiameter / 2)
        neig_x = coor_one[0] - longdiameter / 2 - longdiameter / 4
        NeighbourCoordinates.append([neig_x, neig_y])

    if (coor_one[1] + (shortdiameter / 2) + 1 < y_max) and (
            (coor_one[0] - longdiameter / 2 - longdiameter / 4) - 1 > 0):
        neig_y = coor_one[1] + (shortdiameter / 2)
        neig_x = coor_one[0] - longdiameter / 2 - longdiameter / 4
        NeighbourCoordinates.append([neig_x, neig_y])
    if (coor_one[1] + 2 * shortdiameter + 1 < y_max):  # ?
        neig_y = coor_one[1] + 2 * shortdiameter
        NeighbourCoordinates.append([coor_one[0], neig_y])

    if (coor_one[1] + shortdiameter + shortdiameter / 2 + 1 < y_max) and (
            (coor_one[0] + longdiameter / 2 + longdiameter / 4) + 1 < x_max):
        neig_y = coor_one[1] + shortdiameter + shortdiameter / 2
        neig_x = coor_one[0] + longdiameter / 2 + longdiameter / 4
        NeighbourCoordinates.append([neig_x, neig_y])

    if (coor_one[1] + shortdiameter + 1 < y_max) and (coor_one[0] + (longdiameter / 2 * 3) + 1 < x_max):
        neig_y = coor_one[1] + shortdiameter
        neig_x = coor_one[0] + (longdiameter / 2 * 3)
        NeighbourCoordinates.append([neig_x, neig_y])

    if (coor_one[0] + (longdiameter / 2 * 3) + 1 < x_max):
        neig_x = coor_one[0] + (longdiameter / 2 * 3)
        NeighbourCoordinates.append([neig_x, coor_one[1]])

    if (coor_one[1] - shortdiameter - 1 > 0) and (coor_one[0] + (longdiameter / 2 * 3) + 1 < x_max):
        neig_y = coor_one[1] - shortdiameter
        neig_x = coor_one[0] + (longdiameter / 2 * 3)
        NeighbourCoordinates.append([neig_x, neig_y])

    if (coor_one[1] - shortdiameter - shortdiameter / 2 - 1 > 0) and (
            coor_one[0] + longdiameter / 2 + longdiameter / 4 + 1 < x_max):
        neig_y = coor_one[1] - shortdiameter - shortdiameter / 2
        neig_x = coor_one[0] + longdiameter / 2 + longdiameter / 4
        NeighbourCoordinates.append([neig_x, neig_y])

    if (coor_one[1] - 2 * shortdiameter - 1 > 0):
        neig_y = coor_one[1] - 2 * shortdiameter
        NeighbourCoordinates.append([coor_one[0], neig_y])

    if (coor_one[1] - (3 * shortdiameter / 2) - 1 > 0) and (coor_one[0] - longdiameter / 2 - longdiameter / 4 - 1 > 0):
        neig_y = coor_one[1] - (3 * shortdiameter / 2)
        neig_x = coor_one[0] - longdiameter / 2 - longdiameter / 4
        NeighbourCoordinates.append([neig_x, neig_y])

    if (coor_one[1] - shortdiameter - 1 > 0) and (coor_one[0] - 3 * longdiameter / 2 - 1 > 0):
        neig_y = coor_one[1] - shortdiameter
        neig_x = coor_one[0] - 3 * longdiameter / 2
        NeighbourCoordinates.append([neig_x, neig_y])

    if (coor_one[0] - 3 * longdiameter / 2 - 1 > 0):
        neig_x = coor_one[0] - 3 * longdiameter / 2
        NeighbourCoordinates.append([neig_x, coor_one[1]])

    if (coor_one[1] + shortdiameter + 1 < y_max) and (coor_one[0] - 3 * longdiameter / 2 - 1 > 0):  # ?
        neig_y = coor_one[1] + shortdiameter
        neig_x = coor_one[0] - 3 * longdiameter / 2
        NeighbourCoordinates.append([neig_x, neig_y])

    if (coor_one[1] + shortdiameter + shortdiameter / 2 + 1 < y_max) and (
            coor_one[0] - longdiameter / 2 - longdiameter / 4 - 1 > 0):
        neig_y = coor_one[1] + shortdiameter + shortdiameter / 2
        neig_x = coor_one[0] - longdiameter / 2 - longdiameter / 4
        NeighbourCoordinates.append([neig_x, neig_y])

    for CheckNeigbourSernsors in NeighbourCoordinates:
        NewHexagonID = list(HexagonCoordinate.keys())[
            list(HexagonCoordinate.values()).index([round(item, 2) for item in CheckNeigbourSernsors])]
        if SensorNumberForPrediction[NewHexagonID][SensorType] is not np.nan:
            X_Prediction.append(CheckNeigbourSernsors[0])
            Y_Prediction.append(CheckNeigbourSernsors[1])
            Z_Prediction.append(SensorNumberForPrediction[NewHexagonID][SensorType])
    df = pd.DataFrame({'y_axis': Y_Prediction, 'x_axis': X_Prediction, 'Number of sensors': Z_Prediction})
    # print(df)
    df = df.dropna()
    # print(df)
    X = df[['y_axis', 'x_axis']]
    Y = df['Number of sensors']
    # print(y)
    regr = linear_model.LinearRegression()
    model = regr.fit(X.values, Y.values)
    # print(model.score(X, Y))
    predictedY = int(regr.predict([coor_one]))
    if predictedY < 0:
        predictedY = 0
    # print(f"predicted value of {[coor_one]} is", predictedY)

    return predictedY


def energy_stay(HexagonID):
    energy_stay_value = (4.197 * HexagonHeight[HexagonID] + 275.204) * staying_time(HexagonID)
    return energy_stay_value


def estimated_energy_stay(FutureHexagonID):
    energy_stay_value = (4.197 * HexagonHeight[FutureHexagonID] + 275.204) * estimated_staying_time(FutureHexagonID)
    return energy_stay_value


def estimated_staying_time(FutureHexagonID):
    if episode == 0 and time_step < 5:
        staying_time_value = (4000 / 42 + 6000 / 42) * av_pc_dlr_time
    elif (SensorNumberForPrediction[FutureHexagonID][0] != 'nan') and (
            SensorNumberForPrediction[FutureHexagonID][1] != 'nan'):
        staying_time_value = (SensorNumberForPrediction[FutureHexagonID][0] +
                              SensorNumberForPrediction[FutureHexagonID][1]) * av_pc_dlr_time
    else:
        staying_time_value = (PredictedHexagonSensor(FutureHexagonID, 0) + PredictedHexagonSensor(FutureHexagonID,
                                                                                                  1)) * av_pc_dlr_time
    return staying_time_value


def distance_of_states(HexagonID, NewHexagonID):
    x1, y1 = HexagonCoordinate[HexagonID]
    x2, y2 = HexagonCoordinate[NewHexagonID]
    h1 = HexagonHeight[HexagonID]
    h2 = HexagonHeight[NewHexagonID]
    dist = math.sqrt((y1 - y2) ** 2 + (x1 - x2) ** 2 + (h1 - h2) ** 2)
    return dist


def moving_time(HexagonID, NewHexagonID):
    moving_time_value = distance_of_states(HexagonID, NewHexagonID) / uav_speed
    return moving_time_value


def get_max_value_of_future_state_actions(NewHexagonID):
    list_of_values = []
    for action_index in range(action_number):
        list_of_values.append(q_values[NewHexagonID][action_index])

        # if q_values[NewHexagonID][action_index] == action_is_impossible:
        #     list_of_values.append(action_is_impossible)
        # else:
        #     coor_one = HexagonCoordinate[NewHexagonID]
        #     y_coor = coor_one[1]
        #     x_coor = coor_one[0]
        #     if action_index == 0:
        #         y_coor = coor_one[1] + shortdiameter
        #     elif action_index == 1:
        #         y_coor = coor_one[1] + (shortdiameter / 2)
        #         x_coor = coor_one[0] + longdiameter / 2 + longdiameter / 4
        #     elif action_index == 2:
        #         y_coor = coor_one[1] - shortdiameter / 2
        #         x_coor = coor_one[0] + longdiameter / 2 + longdiameter / 4
        #     elif action_index == 3:
        #         y_coor = coor_one[1] - shortdiameter
        #     elif action_index == 4:
        #         y_coor = coor_one[1] - (shortdiameter / 2)
        #         x_coor = coor_one[0] - longdiameter / 2 - longdiameter / 4
        #     elif action_index == 5:
        #         y_coor = coor_one[1] + (shortdiameter / 2)
        #         x_coor = coor_one[0] - longdiameter / 2 - longdiameter / 4
        #     elif action_index == 6:
        #         y_coor = coor_one[1] + 2 * shortdiameter
        #     elif action_index == 7:
        #         y_coor = coor_one[1] + shortdiameter + shortdiameter / 2
        #         x_coor = coor_one[0] + longdiameter / 2 + longdiameter / 4
        #     elif action_index == 8:
        #         y_coor = coor_one[1] + shortdiameter
        #         x_coor = coor_one[0] + (longdiameter / 2 * 3)
        #     elif action_index == 9:
        #         x_coor = coor_one[0] + (longdiameter / 2 * 3)
        #     elif action_index == 10:
        #         y_coor = coor_one[1] - shortdiameter
        #         x_coor = coor_one[0] + (longdiameter / 2 * 3)
        #     elif action_index == 11:
        #         y_coor = coor_one[1] - shortdiameter - shortdiameter / 2
        #         x_coor = coor_one[0] + longdiameter / 2 + longdiameter / 4
        #     elif action_index == 12:
        #         y_coor = coor_one[1] - 2 * shortdiameter
        #     elif action_index == 13:
        #         y_coor = coor_one[1] - (3 * shortdiameter / 2)
        #         x_coor = coor_one[0] - longdiameter / 2 - longdiameter / 4
        #     elif action_index == 14:
        #         y_coor = coor_one[1] - shortdiameter
        #         x_coor = coor_one[0] - 3 * longdiameter / 2
        #     elif action_index == 15:
        #         x_coor = coor_one[0] - 3 * longdiameter / 2
        #     elif action_index == 16:
        #         y_coor = coor_one[1] + shortdiameter
        #         x_coor = coor_one[0] - 3 * longdiameter / 2
        #     elif action_index == 17:
        #         y_coor = coor_one[1] + shortdiameter + shortdiameter / 2
        #         x_coor = coor_one[0] - longdiameter / 2 - longdiameter / 4
        #     NewHexagonCoordinate = [x_coor, y_coor]
        #     FutureHexagonID = list(HexagonCoordinate.keys())[
        #         list(HexagonCoordinate.values()).index([round(item, 2) for item in NewHexagonCoordinate])]

    return np.max(list_of_values)


def get_next_location(HexagonID, action_index):
    coor_one = HexagonCoordinate[HexagonID]
    y_coor = coor_one[1]
    x_coor = coor_one[0]

    if action_index == 0:
        y_coor = coor_one[1] + shortdiameter
    elif action_index == 1:
        y_coor = coor_one[1] + (shortdiameter / 2)
        x_coor = coor_one[0] + longdiameter / 2 + longdiameter / 4
    elif action_index == 2:
        y_coor = coor_one[1] - shortdiameter / 2
        x_coor = coor_one[0] + longdiameter / 2 + longdiameter / 4
    elif action_index == 3:
        y_coor = coor_one[1] - shortdiameter
    elif action_index == 4:
        y_coor = coor_one[1] - (shortdiameter / 2)
        x_coor = coor_one[0] - longdiameter / 2 - longdiameter / 4
    elif action_index == 5:
        y_coor = coor_one[1] + (shortdiameter / 2)
        x_coor = coor_one[0] - longdiameter / 2 - longdiameter / 4
    elif action_index == 6:
        y_coor = coor_one[1] + 2 * shortdiameter
    elif action_index == 7:
        y_coor = coor_one[1] + shortdiameter + shortdiameter / 2
        x_coor = coor_one[0] + longdiameter / 2 + longdiameter / 4
    elif action_index == 8:
        y_coor = coor_one[1] + shortdiameter
        x_coor = coor_one[0] + (longdiameter / 2 * 3)
    elif action_index == 9:
        x_coor = coor_one[0] + (longdiameter / 2 * 3)
    elif action_index == 10:
        y_coor = coor_one[1] - shortdiameter
        x_coor = coor_one[0] + (longdiameter / 2 * 3)
    elif action_index == 11:
        y_coor = coor_one[1] - shortdiameter - shortdiameter / 2
        x_coor = coor_one[0] + longdiameter / 2 + longdiameter / 4
    elif action_index == 12:
        y_coor = coor_one[1] - 2 * shortdiameter
    elif action_index == 13:
        y_coor = coor_one[1] - (3 * shortdiameter / 2)
        x_coor = coor_one[0] - longdiameter / 2 - longdiameter / 4
    elif action_index == 14:
        y_coor = coor_one[1] - shortdiameter
        x_coor = coor_one[0] - 3 * longdiameter / 2
    elif action_index == 15:
        x_coor = coor_one[0] - 3 * longdiameter / 2
    elif action_index == 16:
        y_coor = coor_one[1] + shortdiameter
        x_coor = coor_one[0] - 3 * longdiameter / 2
    elif action_index == 17:
        y_coor = coor_one[1] + shortdiameter + shortdiameter / 2
        x_coor = coor_one[0] - longdiameter / 2 - longdiameter / 4
    elif action_index == 18:
        y_coor = coor_one[1]
        x_coor = coor_one[0]
    NewHexagonCoordinate = [x_coor, y_coor]
    NewHexagonID = list(HexagonCoordinate.keys())[
        list(HexagonCoordinate.values()).index([round(item, 2) for item in NewHexagonCoordinate])]
    return NewHexagonID


def get_next_action(HexagonID,CurrentUAVNumber):
    ran_num = np.random.random()
    distchecklist = []
    for uavcheck in range(NumberUAVs):
        if CurrentUAVNumber != uavcheck:
            distchecklist.append(ChosenHexagonID[uavcheck])
    if ran_num >= epsilon:
        list_of_values = dict()
        for action_index in range(action_number):
            # print("q_value", q_values[coordinates.index([0,0])][0][0][0])
            if q_values[HexagonID][action_index] != action_is_impossible:
                possible_location = get_next_location(HexagonID, action_index)
                if possible_location not in distchecklist:
                    if any(distance_of_states(possible_location, option) <= UAVsCommDistance for option in distchecklist):
                        list_of_values[action_index]= q_values[HexagonID][action_index]

        MaxValueAction = np.max(list(list_of_values.values()))
        possibleAction = list(list_of_values.keys())[list(list_of_values.values()).index(MaxValueAction)]
        return possibleAction

    else:  # choose a random action
        while True:
            action_num = np.random.randint(action_number)
            # print(f'action number in loop {action_num}')
            if q_values[HexagonID][action_num] != action_is_impossible:
                check_unique = get_next_location(HexagonID, action_num)
                if check_unique not in distchecklist:
                    if any(distance_of_states(check_unique, option) <= UAVsCommDistance for option in distchecklist):
                        return action_num

def q_table_initialize():
    y_max = 973.7
    x_max = 951.5

    for hex_index in range(HexagonNumber):
        coor_one = HexagonCoordinate[hex_index]
        if coor_one[1] + shortdiameter + 1 < y_max:  # ?
            q_values[hex_index][0] = init_q_value
        if (coor_one[1] + (shortdiameter / 2) + 1 < y_max) and (
                (coor_one[0] + longdiameter / 2 + longdiameter / 4) + 1 < x_max):
            q_values[hex_index][1] = init_q_value
        if (coor_one[1] - shortdiameter / 2 - 1 > 0) and (
                (coor_one[0] + longdiameter / 2 + longdiameter / 4) + 1 < x_max):
            q_values[hex_index][2] = init_q_value
        if coor_one[1] - shortdiameter - 1 > 0:
            q_values[hex_index][3] = init_q_value
        if (coor_one[1] - (shortdiameter / 2) - 1 > 0) and (
                (coor_one[0] - longdiameter / 2 - longdiameter / 4) - 1 > 0):
            q_values[hex_index][4] = init_q_value
        if (coor_one[1] + (shortdiameter / 2) + 1 < y_max) and (
                (coor_one[0] - longdiameter / 2 - longdiameter / 4) - 1 > 0):
            q_values[hex_index][5] = init_q_value
        if coor_one[1] + 2 * shortdiameter + 1 < y_max:  # ?
            q_values[hex_index][6] = init_q_value
        if (coor_one[1] + shortdiameter + shortdiameter / 2 + 1 < y_max) and (
                (coor_one[0] + longdiameter / 2 + longdiameter / 4) + 1 < x_max):
            q_values[hex_index][7] = init_q_value
        if (coor_one[1] + shortdiameter + 1 < y_max) and (coor_one[0] + (longdiameter / 2 * 3) + 1 < x_max):
            q_values[hex_index][8] = init_q_value
        if coor_one[0] + (longdiameter / 2 * 3) + 1 < x_max:
            q_values[hex_index][9] = init_q_value
        if (coor_one[1] - shortdiameter - 1 > 0) and (coor_one[0] + (longdiameter / 2 * 3) + 1 < x_max):
            q_values[hex_index][10] = init_q_value
        if (coor_one[1] - shortdiameter - shortdiameter / 2 - 1 > 0) and (
                coor_one[0] + longdiameter / 2 + longdiameter / 4 + 1 < x_max):
            q_values[hex_index][11] = init_q_value
        if coor_one[1] - 2 * shortdiameter - 1 > 0:
            q_values[hex_index][12] = init_q_value
        if (coor_one[1] - (3 * shortdiameter / 2) - 1 > 0) and (
                coor_one[0] - longdiameter / 2 - longdiameter / 4 - 1 > 0):
            q_values[hex_index][13] = init_q_value
        if (coor_one[1] - shortdiameter - 1 > 0) and (coor_one[0] - 3 * longdiameter / 2 - 1 > 0):
            q_values[hex_index][14] = init_q_value
        if coor_one[0] - 3 * longdiameter / 2 - 1 > 0:
            q_values[hex_index][15] = init_q_value
        if (coor_one[1] + shortdiameter + 1 < y_max) and (coor_one[0] - 3 * longdiameter / 2 - 1 > 0):  # ?
            q_values[hex_index][16] = init_q_value
        if (coor_one[1] + shortdiameter + shortdiameter / 2 + 1 < y_max) and (
                coor_one[0] - longdiameter / 2 - longdiameter / 4 - 1 > 0):
            q_values[hex_index][17] = init_q_value
        q_values[hex_index][18] = init_q_value


NumberUAVs = 3
HexagonNumber = 42
Recovery_time1 = 500  # sec
Recovery_time2 = 200
sen_val_max1 = 150
sen_val_max2 = 180
sen_val_min1 = 0
sen_val_min2 = 0
SensorType = 2
shortdiameter = 149.8
longdiameter = 173
UAVsCommDistance = 839
weight_s = 0.8
weight_e = 0.2
uav_speed = 10  # meter/seconds
av_pc_dlr_time = 0.02  # second
theta = 60
total_number_of_sensors = 10000  # 4000 and 6000

SensorNumberForPrediction = dict()
for i in range(HexagonNumber):
    SensorNumberForPrediction[i] = [np.NaN, np.NaN]

sensor_per_cell = [[0 for _ in range(2)] for _ in range(HexagonNumber)]  # number of cells function N()
number_of_sensor_function()
SensingNor = 165660
EnergyNor = 236.3
discount_factor = 0.9  # discount factor for future rewards
learning_rate = 0.01  # the rate at which the AI agent should learn

NumberEpisodes = 10000
epsilon = 0.8
epsilon_min = 0.03
epsilon_decay = 0.9993



InitHexagonID = 0
action_is_impossible = -1000
init_q_value = 0
action_number = 19
actions = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18']
q_values = [[action_is_impossible for _ in range(action_number)] for _ in range(HexagonNumber)]
q_table_initialize()

cum_reward = []
cum_sensing_value = []
cum_reward_per_episode = []
cum_sensing_value_per_episode = []
num_ep_to_plot = []
list_path = [[0] for _ in range(NumberUAVs)]
TimeHistoryOfUAV = [[0] for _ in range(NumberUAVs)]
CumRewarOfEachUAV = [0 for _ in range(NumberUAVs)]
CommonTimeHistory = [0]
AccumulatedValueofLastEpisodeTotal =  [0]

for episode in range(NumberEpisodes):
    res_energy = [250000 for _ in range(NumberUAVs)]
    CurrentTimeUAV = [0 for _ in range(NumberUAVs)]
    prev_covered_time = [0 for _ in range(HexagonNumber)]  # it should be inside of the episode circle
    residual_check = [1 for _ in range(NumberUAVs)]
    accumulated_reward = 0
    accumulated_sensing_value = 0
    time_step = 0
    ChosenHexagonID = [InitHexagonID for _ in range(NumberUAVs)]
    while any(check == 1 for check in residual_check):
        time_list = []
        for i in range(NumberUAVs):
            if residual_check[i] == 1:
                time_list.append(CurrentTimeUAV[i])
            else:
                time_list.append(5000)
        CurrentUAVNumber = np.argmin(time_list)

        CurrentHexagonID = ChosenHexagonID[CurrentUAVNumber]  # store the old row and column indexes
        prev_covered_time[CurrentHexagonID] = CurrentTimeUAV[CurrentUAVNumber]
        action_index_chosen = get_next_action(CurrentHexagonID,CurrentUAVNumber)
        NewHexagonID = get_next_location(CurrentHexagonID, action_index_chosen)
        moving_time_value = moving_time(CurrentHexagonID, NewHexagonID)
        fly_energy = energy_fly(CurrentHexagonID, NewHexagonID)
        estimated_staying_energy = estimated_energy_stay(NewHexagonID)
        energy_to_return_point = energy_fly(NewHexagonID, InitHexagonID)
        if res_energy[CurrentUAVNumber] <= fly_energy + estimated_staying_energy + energy_to_return_point:
            residual_check[CurrentUAVNumber] = 0
            continue
        RewardValue = reward(CurrentHexagonID, NewHexagonID, CurrentTimeUAV[CurrentUAVNumber] + moving_time_value)
        old_q_value = \
            q_values[CurrentHexagonID][action_index_chosen]
        temporal_difference = RewardValue + (
                discount_factor * get_max_value_of_future_state_actions(NewHexagonID=NewHexagonID)) - old_q_value

        new_q_value = old_q_value + (learning_rate * temporal_difference)
        q_values[CurrentHexagonID][action_index_chosen]= new_q_value
        sensing_value = sensing_reward(NewHexagonID, CurrentTimeUAV[CurrentUAVNumber] + moving_time_value)
        accumulated_reward += RewardValue
        accumulated_sensing_value += sensing_value
        CurrentTimeUAV[CurrentUAVNumber] += moving_time_value + staying_time(NewHexagonID)
        SensorNumberForPrediction[NewHexagonID] = HexagonSensor[NewHexagonID]
        staying_energy = energy_stay(NewHexagonID)
        res_energy[CurrentUAVNumber] -= (fly_energy + staying_energy)
        energy_to_return_point = energy_fly(NewHexagonID, InitHexagonID)
        ChosenHexagonID[CurrentUAVNumber] = NewHexagonID
        if episode == NumberEpisodes - 1:
            cum_reward_per_episode.append(RewardValue)
            cum_sensing_value_per_episode.append(sensing_value)
            CumRewarOfEachUAV[CurrentUAVNumber] += RewardValue
            list_path[CurrentUAVNumber].append(NewHexagonID)
            TimeHistoryOfUAV[CurrentUAVNumber].append(CurrentTimeUAV[CurrentUAVNumber])
            CommonTimeHistory.append(CurrentTimeUAV[CurrentUAVNumber])
            AccumulatedValueofLastEpisodeTotal.append(AccumulatedValueofLastEpisodeTotal[-1] + RewardValue)
        time_step += 1

        if res_energy[CurrentUAVNumber] <= energy_to_return_point:
            residual_check[CurrentUAVNumber] = 0
            continue
    if epsilon >= epsilon_min:
        epsilon *= epsilon_decay
    num_ep_to_plot.append(episode)
    cum_sensing_value.append(accumulated_sensing_value)
    cum_reward.append(accumulated_reward)
for ii in range(NumberUAVs):
    list_path[ii].append(0)
    TimeHistoryOfUAV[ii].append(TimeHistoryOfUAV[ii][-1])

eachfifty = []
eachfiftyLine = []
for av in range(0, NumberEpisodes, 50):
    if av + 50 <= len(cum_reward):
        eachfifty.append(av)
        sumrew = 0
        for g in range(50):
            sumrew += cum_reward[av + g]
        eachfiftyLine.append(sumrew / 50)


def write_csv():
    with open('ConventionalSimulationResults.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow("Conventional Total Reward")
        writer.writerow(num_ep_to_plot)
        writer.writerow(cum_reward)
        writer.writerow("Conventional Line")
        writer.writerow(eachfifty)
        writer.writerow(eachfiftyLine)
        writer.writerow("Conventional Sensing Reward")
        writer.writerow(cum_sensing_value)
        writer.writerow("Conventional Total Reward")
        writer.writerow(cum_reward_per_episode)
        writer.writerow("Conventional Sensing Reward")
        writer.writerow(cum_sensing_value_per_episode)
        writer.writerow('UAVs path')
        for i in list_path:
            writer.writerow(i)
        for j in TimeHistoryOfUAV:
            writer.writerow(j)
        writer.writerow("Accumulated Total Reward")
        writer.writerow(CumRewarOfEachUAV)
        writer.writerow(CommonTimeHistory)
        writer.writerow(AccumulatedValueofLastEpisodeTotal)
write_csv()

print('Training complete!')
# print(total_path_list)
# print(cum_sensing_value_per_episode)
# print(cum_reward_per_episode)
# print(cum_sensing_value)
# print(cum_reward)
