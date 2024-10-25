import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random

# Constants
WINDOW_SIZE = 50   # For moving average
Z_THRESHOLD = 3    # Z-score threshold for anomaly detection

# Simulate data stream
def data_stream():
    # Simulate a data stream with seasonal pattern and noise
    while True:
        base_value = np.sin(np.linspace(0, 2 * np.pi, 100))
        seasonal_variation = np.tile(base_value, 10)
        for value in seasonal_variation:
            noise = random.uniform(-0.5, 0.5)
            yield value + noise

# Calculate EMA
def calculate_ema(data, alpha=0.1):
    ema = []
    for i in range(len(data)):
        if i == 0:
            ema.append(data[0])
        else:
            ema.append(alpha * data[i] + (1 - alpha) * ema[i-1])
    return np.array(ema)

# Detect anomalies based on Z-score
def detect_anomalies(data, ema, threshold=Z_THRESHOLD):
    residuals = data - ema
    std_dev = np.std(residuals)
    z_scores = np.abs(residuals / std_dev)
    anomalies = np.where(z_scores > threshold)[0]
    return anomalies

# Real-time visualization
def visualize_real_time():
    # Prepare plot
    plt.ion()
    fig, ax = plt.subplots()
    data_window = deque(maxlen=WINDOW_SIZE)
    ema_window = deque(maxlen=WINDOW_SIZE)
    
    stream = data_stream()
    for i in range(WINDOW_SIZE):
        data_window.append(next(stream))
        ema_window.append(0)

    line, = ax.plot(data_window, label="Data")
    ema_line, = ax.plot(ema_window, label="EMA")
    anomaly_scatter = ax.scatter([], [], color='red', label="Anomalies")
    
    plt.legend()
    plt.title("Real-Time Data Stream with Anomaly Detection")
    plt.xlabel("Time")
    plt.ylabel("Value")
    
    while True:
        data_window.append(next(stream))
        data = np.array(data_window)
        ema = calculate_ema(data)

        # Update plot
        anomalies = detect_anomalies(data, ema)
        line.set_ydata(data)
        ema_line.set_ydata(ema)

        # Update scatter for anomalies
        anomaly_points = np.array([data[i] for i in anomalies])
        anomaly_scatter.set_offsets(np.c_[anomalies, anomaly_points])

        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()

# Run the visualization
visualize_real_time()
