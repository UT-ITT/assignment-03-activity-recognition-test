import csv
import time

import pandas as pd

from DIPPID import SensorUDP

PORT = 5700
DURATION_SECONDS = 10.0

# Configuration
NAME = "sam"
ACTIVITIES = ["running", "rowing", "lifting", "jumpingjacks"]
RECORDINGS_PER_ACTIVITY = 5


def get_output_filename(name, activity, number):
    return f"data/{name}-{activity}-{number}.csv"


def wait_for_button_release(sensor):
    print("Waiting for button_1 press and release to start recording...")

    while True:
        value = sensor.get_value("button_1")
        if value is not None and int(value) == 1:
            break
        time.sleep(0.01)

    while True:
        value = sensor.get_value("button_1")
        if value is not None and int(value) == 0:
            break
        time.sleep(0.01)


def record_for_10_seconds(sensor):
    samples = []
    start = time.time()
    end = start + DURATION_SECONDS

    while time.time() < end:
        acc = sensor.get_value("accelerometer")
        gyro = sensor.get_value("gyroscope")

        if acc and gyro:
            sample = {
                "timestamp": time.time() - start,
                "acc_x": float(acc["x"]),
                "acc_y": float(acc["y"]),
                "acc_z": float(acc["z"]),
                "gyro_x": float(gyro["x"]),
                "gyro_y": float(gyro["y"]),
                "gyro_z": float(gyro["z"]),
            }
            samples.append(sample)

        time.sleep(0.005)

    return samples


def save_and_resample_csv(samples, path):
    headers = ["timestamp", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for s in samples:
            row = [
                round(s["timestamp"], 6),
                round(s["acc_x"], 6),
                round(s["acc_y"], 6),
                round(s["acc_z"], 6),
                round(s["gyro_x"], 6),
                round(s["gyro_y"], 6),
                round(s["gyro_z"], 6),
            ]
            writer.writerow(row)

    # copied from resample.py
    # read csv
    df = pd.read_csv(path)

    # convert timestamps to datetime format
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    # do resample (100Hz - one data point every 10ms)
    df.set_index("timestamp", inplace=True)
    df_resampled = df.resample("10ms").mean()

    # reset index from 1 to n instead of using original indices
    df_resampled.reset_index(inplace=True)

    # convert back to timestamps
    df_resampled["timestamp"] = (
        df_resampled["timestamp"] - pd.Timestamp("1970-01-01")
    ) // pd.Timedelta("1ms")

    # save resampled data to to csv using 'id' as index column name
    df_resampled.index.name = "id"
    df_resampled.to_csv(path)

    return len(df_resampled)


def main():
    sensor = SensorUDP(PORT)
    total_recordings = len(ACTIVITIES) * RECORDINGS_PER_ACTIVITY
    recordings_done = 0

    try:
        for activity in ACTIVITIES:
            for recording_number in range(1, RECORDINGS_PER_ACTIVITY + 1):
                recordings_done += 1
                output_file = get_output_filename(NAME, activity, recording_number)

                print()
                print(f"--- Recording {recordings_done}/{total_recordings} ---")
                print(f"Activity : {activity}")
                print(f"File     : {output_file}")

                wait_for_button_release(sensor)
                print("Recording started...")

                raw_samples = record_for_10_seconds(sensor)
                sample_count = save_and_resample_csv(raw_samples, output_file)

                print("Saved", sample_count, "samples to", output_file)

        print()
        print("All recordings complete!")

    finally:
        sensor.disconnect()


if __name__ == "__main__":
    main()
