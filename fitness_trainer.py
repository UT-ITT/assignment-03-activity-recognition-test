import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import pyglet
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from DIPPID import SensorUDP

ROOT_DIR = "datasets"
IMG_DIR = Path("img")
PORT = 5700
WINDOW_SAMPLES = 1000

ACTIVITY_IMAGES = {
    "jumpingjacks": ["jumpingjack_1.png", "jumpingjack_2.png"],
    "lifting": ["lifting_1.png", "lifting_2.png"],
    "rowing": ["rowing_1.png", "rowing_2.png"],
    "running": ["running_1.png", "running_2.png"],
    "unkown": ["unkown_1.png", "unkown_2.png"],
}

DISPLAY_NAME = {
    "jumpingjacks": "Jumping jacks",
    "lifting": "Lifting",
    "rowing": "Rowing",
    "running": "Running",
    "unkown": "unkown",
}

jumpingjacks_raw = []
lifting_raw = []
rowing_raw = []
running_raw = []
unkown_raw = []

activity_map = {
    "jumpingjacks": jumpingjacks_raw,
    "lifting": lifting_raw,
    "rowing": rowing_raw,
    "running": running_raw,
    "unkown": unkown_raw,
}

for csv_file in Path(ROOT_DIR).rglob("*.csv"):
    filename = csv_file.stem
    user_name, activity, number = filename.split("-")
    df = pd.read_csv(csv_file)
    df = df.dropna()
    activity_map[activity].append(df)


# Alle acc und gyro Spalten werden aufsummiert, um neue Richtungsunabhängige Features zu erhalten.
def compute_sum_features(activity_raw):
    activity_features = []
    for df in activity_raw:
        sum_acc = df[["acc_x", "acc_y", "acc_z"]].abs().to_numpy().sum()
        sum_gyro = df[["gyro_x", "gyro_y", "gyro_z"]].abs().to_numpy().sum()
        activity_features.append((sum_acc, sum_gyro))
    return np.array(activity_features)


def remove_invalid_features(features):
    # Filtert Zeilen mit 0 Einträgen heraus
    valid = (features[:, 0] != 0) & (features[:, 1] != 0)
    return features[valid]


jumpingjacks_valid = remove_invalid_features(compute_sum_features(jumpingjacks_raw))
running_valid = remove_invalid_features(compute_sum_features(running_raw))
rowing_valid = remove_invalid_features(compute_sum_features(rowing_raw))
lifting_valid = remove_invalid_features(compute_sum_features(lifting_raw))

unkown_valid = (
    [[0, 0 + offset] for offset in range(-3000, 16001, 500)]
    + [[500, 0 + offset] for offset in range(-3000, 1, 500)]
    + [[500, 0 + offset] for offset in range(3000, 16001, 500)]
    + [[1000, 0 + offset] for offset in range(-3000, 1, 500)]
    + [[1000, 0 + offset] for offset in range(9000, 16001, 500)]
    + [[1500, 0 + offset] for offset in range(-3000, 1, 500)]
    + [[2000, 0 + offset] for offset in range(-3000, 1, 500)]
    + [[2000, 0 + offset] for offset in range(12000, 16001, 1000)]
    + [[3000, 0 + offset] for offset in range(-3000, 1001, 1000)]
    + [[3000, 0 + offset] for offset in range(13000, 16001, 1000)]
    + [[4000, 0 + offset] for offset in range(-3000, 4001, 1000)]
    + [[4000, 0 + offset] for offset in range(14000, 16001, 1000)]
    + [[5000, 0 + offset] for offset in range(-3000, 6001, 1000)]
    + [[5000, 0 + offset] for offset in range(15000, 16001, 1000)]
    + [[6000, 0 + offset] for offset in range(-3000, 6001, 1000)]
    + [[6000, 0 + offset] for offset in range(15000, 16001, 1000)]
    + [[7000, 0 + offset] for offset in range(-3000, 16001, 1000)]
)

X_parts = []
y_parts = []

for key, arr in [
    ("jumpingjacks", jumpingjacks_valid),
    ("running", running_valid),
    ("rowing", rowing_valid),
    ("lifting", lifting_valid),
    ("unkown", unkown_valid),
]:
    if len(arr) == 0:
        continue
    X_parts.append(arr)
    y_parts.append(np.full(len(arr), key))


X = np.vstack(X_parts)
y = np.concatenate(y_parts)

# train SVM
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Daten skalieren
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# beste Hyperparameter für SVM mit rbf Kernel finden
# -----------------
best_parameters = [[0, 0, 0, 0]]
c_Values = [
    0.01,
    0.1,
    0.5,
    1,
    3,
    5,
    7,
    9,
    10,
    11,
    15,
    20,
    30,
    40,
    100,
]
gamma_Values = [0.0001, 0.001, 0.005, 0.008, 0.01, 0.05, 0.1, 0.8, 1, 2, 3, 5, 10, 100]

# SVM mit Gauß-Kernel (RBF)
for c in c_Values:
    for g in gamma_Values:

        temp_model = svm.SVC(
            kernel="rbf", C=c, gamma=g, probability=True, random_state=42
        )
        temp_model.fit(X_train_scaled, y_train)
        train_score = temp_model.score(X_train_scaled, y_train)
        test_score = temp_model.score(X_test_scaled, y_test)
        if test_score > best_parameters[0][3]:
            best_parameters.clear()
            best_parameters.append([c, g, train_score, test_score])


print(
    f"C: {best_parameters[0][0]}, Gamma: {best_parameters[0][1]} Train_score: {best_parameters[0][2]}, Test_score: {best_parameters[0][3]}"
)
print(best_parameters)
# -------------------

# SVM mit den besten gefundenen Hyperparametern trainieren
classifier = svm.SVC(
    kernel="rbf",
    C=best_parameters[0][0],
    gamma=best_parameters[0][1],
    probability=True,
    random_state=42,
)  # SVM mit Gauß-Kernel (RBF) und den besten Hyperparametern
classifier.fit(X_train_scaled, y_train)

y_pred = classifier.predict(X_test_scaled)

### Confidence

accuracy = accuracy_score(y_test, y_pred)
print("Test accuracy:", round(accuracy, 3))


def scatterplot_two_features(X_plot, y_plot, feature_names, model, title):
    y_arr = np.array(y_plot)
    if np.issubdtype(y_arr.dtype, np.number):
        y_num = y_arr.astype(int)
        classes = np.unique(y_arr)
    else:
        classes, y_num = np.unique(y_arr, return_inverse=True)

    plt.figure(figsize=(10, 6))
    plt.scatter(
        X_plot[:, 0], X_plot[:, 1], c=y_num, cmap="coolwarm", edgecolors="k", s=50
    )

    # Grid für Plot erstellen
    x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
    y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
    nx = 200
    ny = 200
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))

    # Predictions für jeden Punkt im Grid berechnen
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z_pred = model.predict(grid)

    if not np.issubdtype(np.array(Z_pred).dtype, np.number):
        class_to_int = {c: i for i, c in enumerate(classes)}
        Z_num = np.array([class_to_int[z] for z in Z_pred])
    else:
        Z_num = np.array(Z_pred, dtype=int)

    Z = Z_num.reshape(xx.shape)

    plt.contourf(
        xx, yy, Z, alpha=0.2, cmap="coolwarm", levels=np.arange(len(classes) + 1) - 0.5
    )
    plt.colorbar(ticks=range(len(classes)), label="Predicted Class").set_ticklabels(
        classes
    )

    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title(title)
    plt.show()


# live sensor buffer
sample_buffer = deque(maxlen=WINDOW_SAMPLES)


def sensor_value(v):
    if v is None or v == []:
        return None
    return v


def parse_imu_from_sensor(sensor):
    ax = sensor_value(sensor.get_value("acc_x"))
    ay = sensor_value(sensor.get_value("acc_y"))
    az = sensor_value(sensor.get_value("acc_z"))
    gx = sensor_value(sensor.get_value("gyro_x"))
    gy = sensor_value(sensor.get_value("gyro_y"))
    gz = sensor_value(sensor.get_value("gyro_z"))

    if ax is None and sensor.has_capability("accelerometer"):
        acc = sensor_value(sensor.get_value("accelerometer"))
        if isinstance(acc, dict):
            ax, ay, az = acc.get("x"), acc.get("y"), acc.get("z")
        elif isinstance(acc, (list, tuple)) and len(acc) >= 3:
            ax, ay, az = acc[0], acc[1], acc[2]

    if gx is None and sensor.has_capability("gyroscope"):
        gyr = sensor_value(sensor.get_value("gyroscope"))
        if isinstance(gyr, dict):
            gx, gy, gz = gyr.get("x"), gyr.get("y"), gyr.get("z")
        elif isinstance(gyr, (list, tuple)) and len(gyr) >= 3:
            gx, gy, gz = gyr[0], gyr[1], gyr[2]

    if None in (ax, ay, az, gx, gy, gz):
        return None
    try:
        return float(ax), float(ay), float(az), float(gx), float(gy), float(gz)
    except (TypeError, ValueError):
        return None


def window_features_from_buffer(buf):
    if len(buf) < 50:
        return None
    a = np.array(buf, dtype=np.float64)
    if np.isnan(a).any():
        return None
    sum_acc = np.abs(a[:, 0:3]).sum()
    sum_gyro = np.abs(a[:, 3:6]).sum()
    if sum_acc == 0 or sum_gyro == 0:
        return None
    return np.array([[sum_acc, sum_gyro]])


def load_two_sprites(activity_key):
    sprites = []
    for file_name in ACTIVITY_IMAGES[activity_key]:
        path = IMG_DIR / file_name
        image = pyglet.image.load(str(path))
        sprite = pyglet.sprite.Sprite(image, x=40, y=120)
        scale = min(400 / sprite.width, 320 / sprite.height, 1.2)
        sprite.scale = scale
        sprites.append(sprite)
    return sprites


class TrainerWindow(pyglet.window.Window):
    def __init__(self, sensor):
        super().__init__(900, 520, caption="Fitness trainer")
        self.sensor = sensor
        self.last_pred = None
        self.image_frame = 0
        self.last_tick = time.perf_counter()
        self.seconds_per_activity = {k: 0.0 for k in DISPLAY_NAME}

        self.sprites = {}
        for activity_key in DISPLAY_NAME:
            self.sprites[activity_key] = load_two_sprites(activity_key)

        self.pred_label = pyglet.text.Label(
            "",
            font_size=22,
            x=self.width // 2,
            y=70,
            anchor_x="center",
        )
        self.acc_label = pyglet.text.Label(
            "Test accuracy: {:.1%}".format(accuracy),
            font_size=12,
            x=20,
            y=self.height - 40,
        )
        self.time_labels = {}
        y0 = 280
        for i, k in enumerate(DISPLAY_NAME):
            self.time_labels[k] = pyglet.text.Label(
                "",
                font_size=14,
                x=480,
                y=y0 - i * 28,
                anchor_x="left",
            )

        pyglet.clock.schedule_interval(self.tick_sensor, 1.0 / 60.0)
        pyglet.clock.schedule_interval(self.tick_predict, 0.25)
        pyglet.clock.schedule_interval(self.flip_image, 1.0)

    def on_draw(self):
        pyglet.gl.glClearColor(0.5, 0, 0, 1)
        self.clear()
        if self.last_pred is not None:
            self.sprites[self.last_pred][self.image_frame].draw()
        self.pred_label.draw()
        self.acc_label.draw()
        for k, lbl in self.time_labels.items():
            sec = self.seconds_per_activity[k]
            lbl.text = "{}: {:.1f} s".format(DISPLAY_NAME[k], sec)
            lbl.draw()

    def flip_image(self, dt):
        if self.last_pred is not None:
            self.image_frame = 1 - self.image_frame

    def tick_sensor(self, dt):
        row = parse_imu_from_sensor(self.sensor)
        if row is not None:
            sample_buffer.append(row)

        now = time.perf_counter()
        if self.last_pred is not None:
            self.seconds_per_activity[self.last_pred] += now - self.last_tick
        self.last_tick = now

    def tick_predict(self, dt):
        feats = window_features_from_buffer(sample_buffer)
        if feats is None:
            self.pred_label.text = "Collecting sensor…"
            return

        fv = scaler.transform(feats)
        pred = classifier.predict(fv)[0]

        if pred != self.last_pred:
            self.image_frame = 0

        self.last_pred = pred
        self.pred_label.text = "Current: " + DISPLAY_NAME[pred]

    def on_close(self):
        self.sensor.disconnect()
        super().on_close()


def main():
    sensor = SensorUDP(PORT)
    TrainerWindow(sensor)
    pyglet.app.run()


if __name__ == "__main__":
    main()

selected_features = ["Sum Acc", "Sum Gyro"]

scatterplot_two_features(
    X_train_scaled, y_train, selected_features, classifier, "SVM mit Gauß-Kernel"
)
