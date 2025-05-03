import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq, irfft

# Load the sensor data
df = pd.read_csv('raw_signal/piezo_voltage_log-1&3-t1.csv')

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

# === Plot frequency spectrum for each sensor (0–5 Hz) with annotations ===
plt.figure(figsize=(4, 4))
for sensor in sensor_columns:
    signal = df[sensor].values
    frequencies = rfftfreq(len(signal), d=1 / fs)
    fft_magnitude = np.abs(rfft(signal))
    plt.plot(frequencies, fft_magnitude, label=f"{sensor}")

# Add cutoff lines without including in legend
plt.axvline(x=1.5, color='r', linestyle='--')
plt.axvline(x=3.0, color='b', linestyle='--')

# Add annotations for cutoff frequencies
plt.text(1.52, plt.ylim()[1]*0.9, "1.5 Hz\n(90 bpm)", color='r', fontsize=10)
plt.text(3.02, plt.ylim()[1]*0.9, "3.0 Hz\n(180 bpm)", color='b', fontsize=10)

plt.xlim(0, 5)
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
cleaned_data.to_csv('filtered/filtered_piezo_voltage_log-1&3-t1.csv', index=False)
print("Cleaned data saved to 'cleaned_piezo_voltage_log.csv'.")

# === Plot original and cleaned data for each sensor ===
plt.figure(figsize=(10, 8))

for i, sensor in enumerate(sensor_columns, 1):
    ax = plt.subplot(len(sensor_columns), 1, i)
    plt.plot(df['Timestamp_s'], df[sensor], label='Original', alpha=0.5)
    plt.plot(cleaned_data['Timestamp_s'], cleaned_data[sensor], label='Cleaned', color='red')
    plt.title(f'Piezo #{i}')

    # Only show legend on the first subplot
    if i == 1:
        plt.legend()

    # Remove individual x and y labels
    if i != len(sensor_columns):
        ax.set_xlabel('')
    if i != 1:
        ax.set_ylabel('')

# Add shared x and y labels for the whole figure
plt.xlabel('Time (s)', fontsize=12, labelpad=20)
plt.ylabel('Voltage (V)', fontsize=12, labelpad=30)
plt.tight_layout(rect=[0, 0, 1, 0.97])  # leave space for labels
plt.show()


# === Get data ===
time_unfiltered = df['Timestamp_s']
signal_unfiltered = df[sensor].values
time_filtered = cleaned_data['Timestamp_s']
signal_filtered = cleaned_data[sensor].values

# === Frequency domain ===
frequencies = rfftfreq(len(signal_unfiltered), d=1/fs)
fft_unfiltered = np.abs(rfft(signal_unfiltered))
fft_filtered = np.abs(rfft(signal_filtered))

# === Plot ===
fig, axs = plt.subplots(2, 2, figsize=(10, 6))
fig.suptitle(f"Signal Comparison for {sensor}")

# Unfiltered Time Domain
axs[0, 0].plot(time_unfiltered, signal_unfiltered, color='gray')
axs[0, 0].set_title("Unfiltered Signal (Time Domain)")
axs[0, 0].set_xlabel("Time (s)")
axs[0, 0].set_ylabel("Voltage (V)")

# Unfiltered Frequency Domain
axs[0, 1].plot(frequencies, fft_unfiltered, color='black')
axs[0, 1].set_title("Unfiltered Signal (Frequency Domain)")
axs[0, 1].set_xlim(0, 5)
axs[0, 1].set_xlabel("Frequency (Hz)")
axs[0, 1].set_ylabel("Magnitude")

# Filtered Time Domain
axs[1, 0].plot(time_filtered, signal_filtered, color='red')
axs[1, 0].set_title("Filtered Signal (Time Domain)")
axs[1, 0].set_xlabel("Time (s)")
axs[1, 0].set_ylabel("Voltage (V)")

# Filtered Frequency Domain
axs[1, 1].plot(frequencies, fft_filtered, color='darkred')
axs[1, 1].set_title("Filtered Signal (Frequency Domain)")
axs[1, 1].set_xlim(0, 5)
axs[1, 1].set_xlabel("Frequency (Hz)")
axs[1, 1].set_ylabel("Magnitude")

plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
plt.show()
