import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq


def butter_bandpass(data, lowcut, highcut, fs, order=5):
    """
    Design a Butterworth bandpass filter.

    Parameters:
    - lowcut: Lower cutoff frequency in Hz.
    - highcut: Upper cutoff frequency in Hz.
    - fs: Sampling frequency in Hz.
    - order: Filter order. Higher values result in a sharper transition.
    - b, a: Numerator and denominator coefficients of the filter.

    Returns:
    - Filtered data.
    """
    nyq = 0.5 * fs  # Nyquist Frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y


# Load the sensor data
df = pd.read_csv('piezo_voltage_log.csv')

# Define sampling parameters
fs = 1 / np.mean(np.diff(df['Timestamp_s']))  # Sampling frequency in Hz
lowcut = 1  # Lower cutoff frequency in Hz
highcut = 3  # Upper cutoff frequency in Hz
order = 4  # Filter order

# Apply the bandpass filter to each sensor's data
filtered_data = pd.DataFrame()
filtered_data['Timestamp_s'] = df['Timestamp_s']

sensor_columns = [col for col in df.columns if 'Piezo' in col]
for sensor in sensor_columns:
    filtered_data[sensor] = butter_bandpass(df[sensor], lowcut, highcut, fs, order)

# Plot original and filtered data for each sensor
plt.figure(figsize=(12, 8))
for i, sensor in enumerate(sensor_columns, 1):
    plt.subplot(len(sensor_columns), 1, i)
    plt.plot(df['Timestamp_s'], df[sensor], label='Original', alpha=0.5)
    plt.plot(filtered_data['Timestamp_s'], filtered_data[sensor], label='Filtered', color='red')
    plt.title(sensor)
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.legend()
    plt.tight_layout()
plt.show()
