import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random

# Constants
WINDOW_SIZE = 50    # Size of the moving window for real-time display
Z_THRESHOLD = 3     # Threshold Z-score for flagging anomalies

# Simulate data stream with a seasonal pattern and random noise
def data_stream():
    """
    Generates a simulated data stream with a seasonal (sinusoidal) pattern
    and random noise, yielding data points one at a time.
    """
    while True:
        # Base sinusoidal pattern
        base_value = np.sin(np.linspace(0, 2 * np.pi, 100))
        # Repeat this base pattern to extend it over a larger range
        seasonal_variation = np.tile(base_value, 10)
        for value in seasonal_variation:
            # Add random noise to each data point
            noise = random.uniform(-0.5, 0.5)
            yield value + noise

# Calculate Exponential Moving Average (EMA) for a list of data points
def calculate_ema(data, alpha=0.1):
    """
    Computes the Exponential Moving Average (EMA) of a dataset.
    
    Parameters:
    - data (list or array): List of data points.
    - alpha (float): Smoothing factor for EMA. Defaults to 0.1.

    Returns:
    - np.array: EMA values for the dataset.
    """
    ema = []  # List to hold EMA values
    for i in range(len(data)):
        if i == 0:
            # The first EMA value is just the first data point
            ema.append(data[0])
        else:
            # Calculate EMA recursively: alpha * current data + (1 - alpha) * previous EMA
            ema.append(alpha * data[i] + (1 - alpha) * ema[i-1])
    return np.array(ema)

# Identify anomalies in the data based on Z-scores
def detect_anomalies(data, ema, threshold=Z_THRESHOLD):
    """
    Detects anomalies in the data by comparing the residuals to a Z-score threshold.

    Parameters:
    - data (np.array): Original data points.
    - ema (np.array): Calculated EMA for the data points.
    - threshold (float): Z-score threshold above which points are considered anomalies.

    Returns:
    - np.array: Indices of data points classified as anomalies.
    """
    # Calculate residuals (differences) between data and EMA
    residuals = data - ema
    # Compute standard deviation of the residuals
    std_dev = np.std(residuals)
    # Calculate absolute Z-scores for residuals
    z_scores = np.abs(residuals / std_dev)
    # Identify points where Z-score exceeds the threshold
    anomalies = np.where(z_scores > threshold)[0]
    return anomalies

# Real-time visualization of the data stream with EMA and anomalies
def visualize_real_time():
    """
    Visualizes the data stream in real-time with EMA and highlights anomalies.
    Uses matplotlib to dynamically update the plot.
    """
    # Set up real-time plotting
    plt.ion()  # Interactive mode for live updating
    fig, ax = plt.subplots()
    data_window = deque(maxlen=WINDOW_SIZE)  # Rolling window for data points
    ema_window = deque(maxlen=WINDOW_SIZE)   # Rolling window for EMA values

    # Initialize data stream
    stream = data_stream()
    
    # Prime the data window with initial values to start visualization
    for i in range(WINDOW_SIZE):
        data_window.append(next(stream))
        ema_window.append(0)  # Initialize EMA window with zeros for alignment

    # Set up initial lines for data, EMA, and anomalies
    line, = ax.plot(data_window, label="Data")
    ema_line, = ax.plot(ema_window, label="EMA")
    anomaly_scatter = ax.scatter([], [], color='red', label="Anomalies")
    
    # Plot labels and title
    plt.legend()
    plt.title("Real-Time Data Stream with Anomaly Detection")
    plt.xlabel("Time")
    plt.ylabel("Value")
    
    # Continuously update the plot with new data
    while True:
        # Append the next data point from the stream to the rolling window
        data_window.append(next(stream))
        # Convert rolling window to a numpy array for calculations
        data = np.array(data_window)
        # Calculate EMA for the current data window
        ema = calculate_ema(data)

        # Update plot with new data and EMA values
        anomalies = detect_anomalies(data, ema)
        line.set_ydata(data)
        ema_line.set_ydata(ema)

        # Update scatter plot to show anomalies in red
        anomaly_points = np.array([data[i] for i in anomalies])
        anomaly_scatter.set_offsets(np.c_[anomalies, anomaly_points])

        # Rescale plot to fit updated data
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()          # Redraw the figure canvas
        fig.canvas.flush_events()   # Flush GUI events for real-time update

# Start the real-time visualization function
visualize_real_time()
