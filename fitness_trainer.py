# this program visualizes activities with pyglet

import activity_recognizer as activity
import pyglet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


#####
# Read data
#####

ROOT_DIR = "datasets"

jumpingjacks_raw = []
lifting_raw = []
rowing_raw= []
running_raw = []

activity_map = {
    "jumpingjacks": jumpingjacks_raw,
    "lifting": lifting_raw,
    "rowing": rowing_raw,
    "running": running_raw,
}

# Iterate through all folders and CSV files
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

def compute_max_features(activity_raw):
    activity_max = []
    for df in activity_raw:
        max_acc = df[["acc_x", "acc_y", "acc_z"]].abs().to_numpy().max()
        max_gyro = df[["gyro_x", "gyro_y", "gyro_z"]].abs().to_numpy().max()
        activity_max.append((max_acc, max_gyro))

    return np.array(activity_max)


def compute_sum_features(activity_raw):
    activity_features = []
    for df in activity_raw:
        sum_acc = df[["acc_x", "acc_y", "acc_z"]].abs().to_numpy().sum()
        sum_gyro = df[["gyro_x", "gyro_y", "gyro_z"]].abs().to_numpy().sum()
        activity_features.append((sum_acc, sum_gyro))
    return np.array(activity_features)

def compute_std_features(activity_raw):
    activity_features = []
    for df in activity_raw:
        std_acc = df[["acc_x", "acc_y", "acc_z"]].to_numpy().std()
        std_gyro = df[["gyro_x", "gyro_y", "gyro_z"]].to_numpy().std()
        activity_features.append((std_acc, std_gyro))
    return np.array(activity_features)

def compute_mean_features(activity_raw):
    activity_features = []
    for df in activity_raw:
        mean_acc = df[["acc_x", "acc_y", "acc_z"]].to_numpy().mean()
        mean_gyro = df[["gyro_x", "gyro_y", "gyro_z"]].to_numpy().mean()
        activity_features.append((mean_acc, mean_gyro))
    return np.array(activity_features)


def compute_rms_features(activity_raw):
    activity_features = []
    for df in activity_raw:
        acc = df[["acc_x", "acc_y", "acc_z"]].to_numpy()
        gyro = df[["gyro_x", "gyro_y", "gyro_z"]].to_numpy()
        rms_acc = np.sqrt(np.mean(acc ** 2))
        rms_gyro = np.sqrt(np.mean(gyro ** 2))
        activity_features.append((rms_acc, rms_gyro))
    return np.array(activity_features)


def compute_energy_features(activity_raw):
    activity_features = []
    for df in activity_raw:
        acc = df[["acc_x", "acc_y", "acc_z"]].to_numpy()
        gyro = df[["gyro_x", "gyro_y", "gyro_z"]].to_numpy()
        energy_acc = np.sum(acc ** 2)
        energy_gyro = np.sum(gyro ** 2)
        activity_features.append((energy_acc, energy_gyro))
    return np.array(activity_features)


def compute_range_features(activity_raw):
    activity_features = []
    for df in activity_raw:
        acc = df[["acc_x", "acc_y", "acc_z"]].to_numpy()
        gyro = df[["gyro_x", "gyro_y", "gyro_z"]].to_numpy()
        range_acc = acc.max() - acc.min()
        range_gyro = gyro.max() - gyro.min()
        activity_features.append((range_acc, range_gyro))
    return np.array(activity_features)

def compute_rms_energy_features(activity_raw):

    features = []
    for df in activity_raw:
        acc = df[["acc_z"]].to_numpy()
        gyro = df[["gyro_z"]].to_numpy()
        acc_magnitude = np.linalg.norm(acc, axis=1)
        gyro_magnitude = np.linalg.norm(gyro, axis=1)
        rms_acc = np.sqrt(np.mean(acc_magnitude ** 2))
        gyro_energy = np.sum(gyro_magnitude ** 2)
        features.append((rms_acc, gyro_energy))

    return np.array(features)


jumpingjacks_feature = compute_rms_energy_features(jumpingjacks_raw)
running_feature = compute_rms_energy_features(running_raw)
rowing_feature = compute_rms_energy_features(rowing_raw)
lifting_feature = compute_rms_energy_features(lifting_raw)

plt.figure(figsize=(10, 8))
plt.scatter(
    jumpingjacks_feature[:, 0],
    jumpingjacks_feature[:, 1],
    label="Jumping Jacks"
)

plt.scatter(
    running_feature[:, 0],
    running_feature[:, 1],
    label="Running"
)

plt.scatter(
    rowing_feature[:, 0],
    rowing_feature[:, 1],
    label="Rowing"
)

plt.scatter(
    lifting_feature[:, 0],
    lifting_feature[:, 1],
    label="Lifting"
)

plt.xlabel("Max Acceleration")

plt.ylabel("Max Gyroscope")

plt.title("Activity Classification Features")

plt.legend()

plt.grid(True)

plt.show()

