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

print(data_stream())