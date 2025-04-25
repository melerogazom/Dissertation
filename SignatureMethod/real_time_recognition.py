# === Recognition of exercises for the signature method ===

from bleak import BleakClient
import asyncio
import struct
import signal
import sys
import numpy as np
import esig  # Signature transform library (feature extraction technique)
import csv
import threading
import os
import time
import joblib

# Configuration
IMU_ADDRESS = "E6C97A8E-59A4-4ED8-B539-1EDE4EA69603" # MAC address of the IMU device
CHARACTERISTIC_UUID = "0000ffe4-0000-1000-8000-00805f9a34fb" # BLE characteristic to read data from
SIG_LEVEL = 3  # Signature truncation level (degree of signature transform)
MIN_WINDOW_SIZE = 15  # Minimum IMU data window for feature extraction
CSV_FILE = "signature_training_data.csv" # CSV file for storing features and labels

# Globals 
buffer = [] # Rolling buffer for IMU data points
latest_segment = [] # Most recent segment used for feature extraction
running = True # Flag to control the main loop
current_exercise = ""  # Track current exercise type
# clf = None  # Will hold the classifier for the current exercise

# handle to exit
def signal_handler(sig, frame):
    global running
    print("\nStopping logger...")
    running = False

signal.signal(signal.SIGINT, signal_handler)

# compute Signature Features
# This is where the ESIG library transforms the time-series IMU data into a high-dimensional feature vector.
def compute_signature(data):
    path = np.array(data) 
    return esig.stream2sig(path, SIG_LEVEL) # ESIG transform up to SIG_LEVEL (higher = richer features)

# Save to CSV 
def write_to_csv(signature, label, exercise):
    sig_flattened = [f"{x:.4f}" for x in signature.flatten()] # Format features nicely
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S") # Human-readable timestamp
    file_exists = os.path.isfile(CSV_FILE)

    with open(CSV_FILE, "a", newline='') as f:
        writer = csv.writer(f)

        # Write header if the file is new or empty
        if not file_exists or os.stat(CSV_FILE).st_size == 0:
            header = ["Timestamp"] + [f"Sig_{i+1}" for i in range(len(sig_flattened))] + ["Exercise_Type", "Label"]
            writer.writerow(header)

        writer.writerow([timestamp] + sig_flattened + [exercise, label])

    print(f"Signature saved at {timestamp} with label {label}")

# Manual input thread 
# Handles user interaction: start capture, predict, label, save
def user_input_thread():
    global buffer, latest_segment, current_exercise, clf

    while running:
        current_exercise = input("Enter the exercise type (e.g., bicep_curl, lateral_raise, front_raise, tricep_pulldown): ").strip()

        # Load the classifier model for the current exercise
        # try:
        #     clf = joblib.load(f'activity_classifier_{current_exercise}.pkl')
        #     print(f"Loaded classifier for {current_exercise}")
        # except Exception as e:
        #     print(f"Error loading classifier for {current_exercise}: {e}")
        #     continue

        # Wait for user to start capture
        input("Press Enter when you're ready to START capturing movement...")
        print("Capturing movement window...")
        time.sleep(3) # Short delay to ensure IMU data is collected

        # Ensure enough data is available
        if len(buffer) < MIN_WINDOW_SIZE:
            print("Not enough data yet. Wait a few more seconds of movement.")
            continue

        # Take the latest window of data
        latest_segment = buffer.copy()[-MIN_WINDOW_SIZE:]
        sig = compute_signature(latest_segment)
        sig_flattened = np.array(sig).flatten().reshape(1, -1)

        # Make prediction using the loaded classifier
        # if clf:
        #     prediction = clf.predict(sig_flattened)[0]
        #     print(f"Real-time prediction: {'REP' if prediction == 1 else 'NON-REP'}")

        # Manual labeling
        label = int(input("Enter true label [1=rep, 0=non-rep]: ").strip())
        write_to_csv(sig, label, current_exercise)
        print("Signature saved.")

# IMU data callback handler
# Converts raw BLE packet bytes into real IMU measurements.
def process_imu_data(sender, data):
    global buffer
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

            # Keep buffer size within limit
            if len(buffer) > MIN_WINDOW_SIZE:
                buffer.pop(0) 
    except Exception as e:
        print(f"IMU data error: {e}")

# Main
async def main():
    print(f"Connecting to IMU at {IMU_ADDRESS}")
    try:
        async with BleakClient(IMU_ADDRESS) as client:
            print("Connected. Start doing reps when ready.")
            await client.start_notify(CHARACTERISTIC_UUID, process_imu_data)

            # Start separate thread to handle user interaction
            input_thread = threading.Thread(target=user_input_thread)
            input_thread.daemon = True
            input_thread.start()

            # Keep this loop alive while running is True
            while running:
                await asyncio.sleep(0.1)

            await client.stop_notify(CHARACTERISTIC_UUID)
            print("Disconnected.")

    except Exception as e:
        print(f"BLE connection error: {e}")

# Run
if __name__ == "__main__":
    # Start the main BLE event loop
    asyncio.run(main())
    print("Session complete.")