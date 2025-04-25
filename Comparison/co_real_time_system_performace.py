import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import asyncio
import threading
import struct
import time
from bleak import BleakClient
import tkinter as tk
from tkinter import ttk
from tkinter import font
from tkinter import Canvas

# ==== Configuration ====
IMU_ADDRESS = "E6C97A8E-59A4-4ED8-B539-1EDE4EA69603"  
CHARACTERISTIC_UUID = "0000ffe4-0000-1000-8000-00805f9a34fb"  
MIN_WINDOW_SIZE = 15

# ==== Globals ====
buffer = []
data_points = []
running = True
clf = joblib.load("exercise_type_classifier_handcrafted.pkl")  # Handcrafted multi-class model
prediction_label = None
rep_count_label = None
gyro_canvas = None
accel_canvas = None
rep_counts = {}  # Rep counter for each exercise

# === Handcrafted Feature Computation ===
def compute_handcrafted_features(segment):
    arr = np.array(segment)  
    features = []
    
    for i in range(arr.shape[1]):
        axis = arr[:, i]
        mean = np.mean(axis)
        std = np.std(axis)
        min_val = np.min(axis)
        max_val = np.max(axis)
        rng = max_val - min_val
        rms = np.sqrt(np.mean(axis**2))
        energy = np.sum(axis**2) / len(axis)
        zcr = ((axis[:-1] * axis[1:]) < 0).sum()
        
        features += [mean, std, min_val, max_val, rng, rms, energy, zcr]
    
    sma = np.mean(np.sum(np.abs(arr[:, :3]), axis=1))  # SMA of acc
    features.append(sma)
    
    return np.array(features)

def is_motion_detected(segment):
    arr = np.array(segment)
    std = np.std(arr, axis=0)
    return np.mean(std) > 0.2

# === BLE Data Handler ===
def process_imu_data(sender, data):
    global buffer, data_points
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

# === Controlled Prediction Trigger ===
def trigger_detection():
    countdown_and_predict(3)

def countdown_and_predict(count):
    if count > 0:
        prediction_label.config(text=f"Get ready... {count}")
        prediction_label.after(1000, lambda: countdown_and_predict(count - 1))
    else:
        perform_prediction()

def perform_prediction():
    global buffer, rep_counts

    if len(buffer) >= MIN_WINDOW_SIZE:
        segment = buffer.copy()[-MIN_WINDOW_SIZE:]
        if not is_motion_detected(segment):
            prediction_label.config(text="No motion detected. Try again.", fg="#FF4500")
            return

        features = compute_handcrafted_features(segment)
        features_flat = features.reshape(1, -1)
        prediction = clf.predict(features_flat)[0]

        rep_counts[prediction] = rep_counts.get(prediction, 0) + 1

        prediction_label.config(text=f"Exercise: {prediction}", fg="#00FF00")
        rep_count_label.config(text=f"Reps: {rep_counts[prediction]}")

        update_visuals()

# === GUI ===
def start_interface():
    global prediction_label, rep_count_label, gyro_canvas, accel_canvas

    root = tk.Tk()
    root.title("Live Exercise Recognition (Handcrafted Features)")
    root.geometry("500x750")
    root.configure(bg="#1e1e1e")

    rep_count_label = tk.Label(root, text="Reps: 0", font=("Helvetica", 24, "bold"), bg="#1e1e1e", fg="white")
    rep_count_label.pack(pady=10)

    prediction_label = tk.Label(root, text="Waiting for prediction...", font=("Helvetica", 24, "bold"), bg="#1e1e1e", fg="white")
    prediction_label.pack(pady=10)

    start_button = tk.Button(root, text="Start Rep Detection", font=("Helvetica", 16, "bold"), command=trigger_detection)
    start_button.pack(pady=10)

    ttk.Label(root, text="Gyroscope Data:", foreground="white", background="#1e1e1e", font=("Helvetica", 16, "bold")).pack()
    gyro_canvas = Canvas(root, width=500, height=200, bg="black")
    gyro_canvas.pack(pady=10)

    ttk.Label(root, text="Acceleration Data:", foreground="white", background="#1e1e1e", font=("Helvetica", 16, "bold")).pack()
    accel_canvas = Canvas(root, width=500, height=200, bg="black")
    accel_canvas.pack(pady=10)

    root.mainloop()

# === Visualization Update ===
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

# === BLE Thread ===
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