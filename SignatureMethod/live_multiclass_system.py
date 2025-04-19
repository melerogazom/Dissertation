import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import asyncio
import threading
import struct
import time
import os
from bleak import BleakClient
from esig import stream2sig
import tkinter as tk
from tkinter import ttk
from tkinter import font
from tkinter import Canvas

# ==== Configuration ====
IMU_ADDRESS = "E6C97A8E-59A4-4ED8-B539-1EDE4EA69603"
CHARACTERISTIC_UUID = "0000ffe4-0000-1000-8000-00805f9a34fb"
SIG_LEVEL = 3
MIN_WINDOW_SIZE = 15

# ==== Globals ====
buffer = []
latest_segment = []
running = True
current_exercise = "lateral_raise"
clf = joblib.load(f'activity_classifier_{current_exercise}.pkl')
prediction_label = None
rep_count_label = None
exercise_label = None
rep_count = 0
last_prediction = 0
gyro_canvas = None
accel_canvas = None
data_points = []
collecting_data = True
can_predict = True
cooldown_active = False
ready_to_collect = True
rep_detected_time = 0
rep_timeout = 1.2  # in seconds


def compute_signature(data):
    path = np.array(data)
    return stream2sig(path, SIG_LEVEL)

def process_imu_data(sender, data):
    global buffer, data_points, collecting_data, ready_to_collect
    if not collecting_data or not ready_to_collect:
        return

    try:
        if len(data) >= 18:
            ax = struct.unpack('<h', data[0:2])[0] / 32768.0 * 16
            ay = struct.unpack('<h', data[2:4])[0] / 32768.0 * 16
            az = struct.unpack('<h', data[4:6])[0] / 32768.0 * 16
            gx = struct.unpack('<h', data[6:8])[0] / 32768.0 * 2000
            gy = struct.unpack('<h', data[8:10])[0] / 32768.0 * 2000
            gz = struct.unpack('<h', data[10:12])[0] / 32768.0 * 2000
            roll = struct.unpack('<h', data[12:14])[0] / 32768.0 * 180
            pitch = struct.unpack('<h', data[14:16])[0] / 32768.0 * 180
            yaw = struct.unpack('<h', data[16:18])[0] / 32768.0 * 180

            point = [ax, ay, az, gx, gy, gz, roll, pitch, yaw]
            buffer.append(point)
            data_points.append((gx, gy, gz, ax, ay, az))

            if len(buffer) > MIN_WINDOW_SIZE:
                buffer.pop(0)
            if len(data_points) > 100:
                data_points.pop(0)
    except Exception as e:
        print(f"⚠️ IMU data error: {e}")

exercise_clf = joblib.load("exercise_type_classifier.pkl")

def reset_after_cooldown():
    global collecting_data, cooldown_active, can_predict, buffer, data_points, ready_to_collect
    # Clear all data more aggressively
    buffer.clear()
    data_points.clear()
    # Add a small delay before starting to collect again
    time.sleep(0.5)
    collecting_data = True
    can_predict = True
    cooldown_active = False
    ready_to_collect = True
    prediction_label.config(text="Ready for next rep 🔄", fg="white")

def clear_all_data():
    global buffer, data_points, collecting_data, ready_to_collect
    buffer.clear()
    data_points.clear()
    collecting_data = False
    ready_to_collect = False
    time.sleep(0.5)  # Add a small delay
    collecting_data = True
    ready_to_collect = True

def make_prediction():
    global prediction_label, rep_count, last_prediction, current_exercise, exercise_label
    global cooldown_active, collecting_data, can_predict, buffer, data_points, ready_to_collect, rep_detected_time

    if cooldown_active or not can_predict or not ready_to_collect:
        prediction_label.after(100, make_prediction)
        update_visuals()
        return

    if len(buffer) >= MIN_WINDOW_SIZE:
        latest_segment = buffer.copy()[-MIN_WINDOW_SIZE:]
        sig = compute_signature(latest_segment)
        sig_flattened = np.array(sig).flatten().reshape(1, -1)

        clf_path = f'activity_classifier_{current_exercise}.pkl'
        if os.path.exists(clf_path):
            clf = joblib.load(clf_path)
            rep_pred = clf.predict(sig_flattened)[0]

            now = time.time()

            if rep_pred == 1 and last_prediction == 0 and now - rep_detected_time > rep_timeout:
                rep_detected_time = now
                rep_count += 1
                rep_count_label.config(text=f"Reps: {rep_count}")

                # Create DataFrame with feature names for exercise classifier
                feature_names = [f'Sig_{i+1}' for i in range(sig_flattened.shape[1])]
                sig_df = pd.DataFrame(sig_flattened, columns=feature_names)
                exercise_pred = exercise_clf.predict(sig_df)[0]
                
                current_exercise = exercise_pred
                exercise_label.config(text=f"Last Exercise: {exercise_pred}")

                prediction_label.config(text=f"{exercise_pred}: REP 💪", fg="#00FF00")

                # Use the new clear_all_data function instead of direct buffer clearing
                clear_all_data()
                cooldown_active = True
                can_predict = False

                prediction_label.after(int(rep_timeout * 1000), reset_after_cooldown)

            elif rep_pred == 0:
                prediction_label.config(text="Waiting for rep...", fg="#FF4500")

            last_prediction = rep_pred

    prediction_label.after(100, make_prediction)
    update_visuals()

def update_visuals():
    global gyro_canvas, accel_canvas, data_points
    if gyro_canvas is None or accel_canvas is None:
        return

    gyro_canvas.delete("all")
    accel_canvas.delete("all")

    width, height = 500, 200

    gyro_canvas.create_text(250, 10, text="Gyroscope Data (°/s)", fill="white", font=("Helvetica", 14, "bold"))
    for i, (gx, gy, gz, _, _, _) in enumerate(data_points):
        x = i * (width / len(data_points))
        gyro_canvas.create_line(x, height/2 + (gx / 2000.0) * 100, x, height/2, fill="#00FFFF")
        gyro_canvas.create_line(x, height/2 + (gy / 2000.0) * 100, x, height/2, fill="#FF69B4")
        gyro_canvas.create_line(x, height/2 + (gz / 2000.0) * 100, x, height/2, fill="#FFD700")

    accel_canvas.create_text(250, 10, text="Acceleration Data (m/s²)", fill="white", font=("Helvetica", 14, "bold"))
    for i, (_, _, _, ax, ay, az) in enumerate(data_points):
        x = i * (width / len(data_points))
        accel_canvas.create_line(x, height/2 + (ax / 16.0) * 100, x, height/2, fill="#7FFF00")
        accel_canvas.create_line(x, height/2 + (ay / 16.0) * 100, x, height/2, fill="#FF4500")
        accel_canvas.create_line(x, height/2 + (az / 16.0) * 100, x, height/2, fill="#9400D3")

def start_interface():
    global prediction_label, rep_count_label, gyro_canvas, accel_canvas, exercise_label

    root = tk.Tk()
    root.title("Live Exercise Recognition")
    root.geometry("500x700")
    root.configure(bg="#1e1e1e")

    rep_count_label = tk.Label(root, text="Reps: 0", font=("Helvetica", 24, "bold"), bg="#1e1e1e", fg="white")
    rep_count_label.pack(pady=10)

    exercise_label = tk.Label(root, text="Last Exercise: None", font=("Helvetica", 18), bg="#1e1e1e", fg="white")
    exercise_label.pack(pady=5)

    prediction_label = tk.Label(root, text="Waiting for prediction...", font=("Helvetica", 24, "bold"), bg="#1e1e1e", fg="white")
    prediction_label.pack(pady=10)

    ttk.Label(root, text="Gyroscope Data:", foreground="white", background="#1e1e1e", font=("Helvetica", 16, "bold")).pack()
    gyro_canvas = Canvas(root, width=500, height=200, bg="black")
    gyro_canvas.pack(pady=10)

    ttk.Label(root, text="Acceleration Data:", foreground="white", background="#1e1e1e", font=("Helvetica", 16, "bold")).pack()
    accel_canvas = Canvas(root, width=500, height=200, bg="black")
    accel_canvas.pack(pady=10)

    make_prediction()
    root.mainloop()

def run_ble():
    async def run():
        async with BleakClient(IMU_ADDRESS) as client:
            await client.start_notify(CHARACTERISTIC_UUID, process_imu_data)
            while running:
                await asyncio.sleep(0.1)

    asyncio.run(run())

t = threading.Thread(target=run_ble)
t.daemon = True
t.start()

start_interface()

"""
Gyroscope Graph:
- X-Axis (°/s) - Cyan (#00FFFF)
- Y-Axis (°/s) - Pink (#FF69B4)
- Z-Axis (°/s) - Gold (#FFD700)

Acceleration Graph:
- X-Axis (m/s²) - Light Green (#7FFF00)
- Y-Axis (m/s²) - Orange-Red (#FF4500)
- Z-Axis (m/s²) - Dark Violet (#9400D3)
"""
