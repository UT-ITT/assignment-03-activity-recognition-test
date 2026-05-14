# this program visualizes activities with pyglet

import activity_recognizer as activity
import pyglet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from pathlib import Path

#####
# Read data
#####

ROOT_DIR = "datasets"

jumpingjacks_raw = []
lifting_raw = []
rowing_raw = []
running_raw = []

activity_map = {
    "jumpingjacks": jumpingjacks_raw,
    "lifting": lifting_raw,
    "rowing": rowing_raw,
    "running": running_raw,
}

for csv_file in Path(ROOT_DIR).rglob("*.csv"):

    filename = csv_file.stem
    try:
        user_name, activity, number = filename.split("-")
    except ValueError:
        print(f"Skipping invalid filename: {csv_file.name}")
        continue

    df = pd.read_csv(csv_file)

    if activity in activity_map:
        activity_map[activity].append(df)
    else:
        print(f"Unknown activity '{activity}' in file: {csv_file.name}")

#####
# calculate features
#####


def compute_sum_features(activity_raw):
    activity_features = []
    for df in activity_raw:
        sum_acc = df[["acc_x", "acc_y", "acc_z"]].abs().to_numpy().sum()
        sum_gyro = df[["gyro_x", "gyro_y", "gyro_z"]].abs().to_numpy().sum()
        activity_features.append((sum_acc, sum_gyro))
    return np.array(activity_features)


jumpingjacks_feature = compute_sum_features(jumpingjacks_raw)
running_feature = compute_sum_features(running_raw)
rowing_feature = compute_sum_features(rowing_raw)
lifting_feature = compute_sum_features(lifting_raw)


#####
# clean
#####


def remove_invalid_features(invalid_features):
    valid_mask = (invalid_features[:, 0] != 0) & (invalid_features[:, 1] != 0)
    return invalid_features[valid_mask]


jumpingjacks_valid = remove_invalid_features(jumpingjacks_feature)
running_valid = remove_invalid_features(running_feature)
rowing_valid = remove_invalid_features(rowing_feature)
lifting_valid = remove_invalid_features(lifting_feature)


#####
# plot features
#####

plt.figure(figsize=(10, 8))
plt.scatter(jumpingjacks_valid[:, 0], jumpingjacks_valid[:, 1], label="Jumping Jacks")
plt.scatter(running_valid[:, 0], running_valid[:, 1], label="Running")
plt.scatter(rowing_valid[:, 0], rowing_valid[:, 1], label="Rowing")
plt.scatter(lifting_valid[:, 0], lifting_valid[:, 1], label="Lifting")
plt.xlabel("Sum Acceleration")
plt.ylabel("Sum Gyroscope")
plt.title("Activity Classification Features")
plt.legend()
plt.grid(True)
plt.show()
