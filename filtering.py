import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq, irfft

# Load the sensor data
df = pd.read_csv('piezo_voltage_log-1&3-t1.csv')

def butter_bandpass(data, lowcut, highcut, fs, order=5):
    """Apply a Butterworth bandpass filter to the input signal."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


# Sampling parameters
fs = 1 / np.mean(np.diff(df['Timestamp_s']))
lowcut = 1.5
highcut = 3.0
order = 4

# Sensor columns
sensor_columns = [col for col in df.columns if 'Piezo' in col]

# === Plot frequency spectrum for each sensor ===
plt.figure(figsize=(12, 6))
for sensor in sensor_columns:
    signal = df[sensor].values
    frequencies = rfftfreq(len(signal), d=1 / fs)
    fft_magnitude = np.abs(rfft(signal))
    plt.plot(frequencies, fft_magnitude, label=f"{sensor}")

plt.axvline(x=1.5, color='r', linestyle='--', label="1.5 Hz / 90 bpm Cutoff")
plt.axvline(x=3.0, color='b', linestyle='--', label="3.0 Hz / 180 bpm Cutoff")

plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Frequency Spectrum of Piezo Signals")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Apply bandpass (+ magnitude threshold) filtering ===
cleaned_data = pd.DataFrame()
cleaned_data['Timestamp_s'] = df['Timestamp_s']

for sensor in sensor_columns:
    # Bandpass filter
    bandpassed = butter_bandpass(df[sensor].values, lowcut, highcut, fs, order)

    # FFT
    fft_vals = rfft(bandpassed)
    fft_magnitude = np.abs(fft_vals)

    fft_vals[fft_magnitude < 0] = 0

    # Inverse FFT to reconstruct cleaned signal
    cleaned_signal = irfft(fft_vals, n=len(bandpassed))
    cleaned_data[sensor] = cleaned_signal

# Save cleaned data
cleaned_data.to_csv('filtered_piezo_voltage_log-1&3-t1.csv', index=False)
print("Cleaned data saved to 'cleaned_piezo_voltage_log.csv'.")

# === Plot original and cleaned data for each sensor ===
plt.figure(figsize=(12, 8))
for i, sensor in enumerate(sensor_columns, 1):
    plt.subplot(len(sensor_columns), 1, i)
    plt.plot(df['Timestamp_s'], df[sensor], label='Original', alpha=0.5)
    plt.plot(cleaned_data['Timestamp_s'], cleaned_data[sensor], label='Cleaned', color='red')
    plt.title(sensor)
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.legend()
    plt.tight_layout()
plt.show()
