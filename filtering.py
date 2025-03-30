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
fs = 1 / np.mean(np.diff(df['Timestamp_s']))  # Sampling frequency in Hz = 1/ average time step
lowcut = 2.2  # Lower cutoff frequency in Hz 1.87 for 110bpm
highcut = 2.67  # Upper cutoff frequency in Hz
order = 4  # Filter order

# Get sensor column names
sensor_columns = [col for col in df.columns if 'Piezo' in col]

# Plot frequency spectrum for each sensor
plt.figure(figsize=(12, 6))

for sensor in sensor_columns:
    signal = df[sensor].values
    frequencies = np.fft.rfftfreq(len(signal), d=1 / fs)
    fft_magnitude = np.abs(np.fft.rfft(signal))

    plt.plot(frequencies, fft_magnitude, label=f"{sensor}")

# Add cutoff lines
plt.axvline(x=2.2, color='r', linestyle='--', label="2.2 Hz Cutoff")
plt.axvline(x=2.67, color='b', linestyle='--', label="2.67 Hz Cutoff")

plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Frequency Spectrum of Piezo Signals")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Apply the bandpass filter to each sensor's data
filtered_data = pd.DataFrame()
filtered_data['Timestamp_s'] = df['Timestamp_s']

for sensor in sensor_columns:
    filtered_data[sensor] = butter_bandpass(df[sensor], lowcut, highcut, fs, order)

# Save the filtered data to CSV
filtered_data.to_csv('filtered_piezo_voltage_log.csv', index=False)
print("Filtered data saved to 'filtered_piezo_voltage_log.csv'.")


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
