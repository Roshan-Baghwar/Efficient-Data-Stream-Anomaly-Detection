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

arr = [1,2,3,4,5]
ema = calculate_ema(arr)
print(arr, ema)