import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_labeled_temperature_data(num_samples=259200, noise_level=0.5, 
                                        daily_anomaly_fraction=0.02, seasonal_anomaly_fraction=0.1,
                                        downsample_rate=1):
    """
    Generates synthetic temperature data with both daily and seasonal variations,
    and injects anomalies. Optionally downsample the data.
    
    Parameters:
      num_samples (int): Total number of data points (default for 6 months = 259200).
      noise_level (float): Std. dev. of Gaussian noise.
      daily_anomaly_fraction (float): Fraction of points for daily anomalies.
      seasonal_anomaly_fraction (float): Fraction of days with seasonal anomaly.
      downsample_rate (int): Take one reading every 'downsample_rate' points.
      
    Returns:
      temperature (np.ndarray): Synthetic temperature readings.
      labels (np.ndarray): Labels (0 for normal, 1 for anomaly).
    """
    t = np.arange(num_samples)
    
    # Daily cycle: period = 1440 samples (one day)
    daily_cycle = 10 * np.sin(2 * np.pi * t / 1440) + 25
    
    # Seasonal cycle: slower trend across the whole simulation
    seasonal_cycle = 5 * np.sin(2 * np.pi * t / num_samples)
    
    noise = np.random.normal(0, noise_level, num_samples)
    
    temperature = daily_cycle + seasonal_cycle + noise
    labels = np.zeros(num_samples, dtype=int)
    
    # Inject daily anomalies (short spikes/dips)
    num_daily_anomalies = int(daily_anomaly_fraction * num_samples)
    daily_anomaly_indices = np.random.choice(num_samples, num_daily_anomalies, replace=False)
    for idx in daily_anomaly_indices:
        spike = np.random.choice([15, -15])
        temperature[idx] += spike
        labels[idx] = 1
        
    # Inject seasonal anomalies (sustained shifts for a full day)
    num_days = num_samples // 1440
    num_seasonal_anomalies = int(seasonal_anomaly_fraction * num_days)
    seasonal_anomaly_days = np.random.choice(np.arange(num_days), num_seasonal_anomalies, replace=False)
    for day in seasonal_anomaly_days:
        start_idx = day * 1440
        end_idx = start_idx + 1440
        shift = np.random.choice([5, -5])
        temperature[start_idx:end_idx] += shift
        labels[start_idx:end_idx] = 1

    # Downsample the data if needed
    if downsample_rate > 1:
        temperature = temperature[::downsample_rate]
        labels = labels[::downsample_rate]
    
    return temperature, labels

def create_windows_and_labels(data, labels, window_size=60):
    """
    Splits the data and corresponding labels into overlapping windows.
    A window is labeled as anomalous (1) if any sample in it is anomalous.
    
    Parameters:
      data (np.ndarray): 1D temperature data.
      labels (np.ndarray): 1D labels (0/1) for each data point.
      window_size (int): Number of time steps per window.
      
    Returns:
      windows (np.ndarray): 2D array with each row as a window.
      window_labels (np.ndarray): 1D array of window-level labels.
    """
    windows = []
    window_labels = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i+window_size]
        windows.append(window)
        window_label = 1 if np.any(labels[i:i+window_size] == 1) else 0
        window_labels.append(window_label)
    return np.array(windows), np.array(window_labels)

if __name__ == '__main__':
    # For example, downsample by a factor of 5 (i.e., one reading every 5 minutes)
    data, labels = generate_labeled_temperature_data(num_samples=259200, noise_level=0.5, 
                                                       daily_anomaly_fraction=0.02, seasonal_anomaly_fraction=0.1,
                                                       downsample_rate=5)
    print("Original samples:", 259200, "Downsampled samples:", len(data))
    
    # Save raw data for presentation
    df = pd.DataFrame({
        'time': np.arange(len(data)),
        'temperature': data,
        'label': labels
    })
    df.to_csv("synthetic_temperature_data_downsampled.csv", index=False)
    
    # Plot data for a quick visualization
    plt.figure(figsize=(12, 4))
    plt.plot(data, label="Temperature")
    plt.scatter(np.where(labels==1), data[labels==1], color='red', label="Anomaly", s=10)
    plt.xlabel("Time (downsampled index)")
    plt.ylabel("Temperature (°C)")
    plt.title("Downsampled Synthetic Temperature Data with Anomalies")
    plt.legend()
    plt.show()
