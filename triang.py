import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture

# === Setup ===
from sklearn.preprocessing import StandardScaler

FILENAME = 'filtered_piezo_voltage_log-1&1-t1.csv'
SPEED_OF_SOUND = 162900  # cm/s in hydrogel

# Sensor coordinates (centimeters)
sensor_coords = np.array([
    [9.0, 8.5],
    [11.7, 15.5],
    [3.5, 2.5],
    [2.5, 11.5],
    [5.0, 18.5],
])

speaker_coords = np.array([
    [7.0, 16.0],
    [7.0, 6.0],
])

# === Step 1: Load filtered data ===
df = pd.read_csv(FILENAME)
time = df['Timestamp_s'].values  # keep in seconds
signals = df[[col for col in df.columns if 'Piezo' in col]].values
NUM_SENSORS = signals.shape[1]

# === Step 2: Detect all peaks per sensor ===
all_peaks = []
peak_records = []

for i in range(NUM_SENSORS):
    signal = signals[:, i]
    gradient = np.diff(signal)

    # Peak: rising then falling (local max)
    peaks = np.where((gradient[:-1] > 0) & (gradient[1:] <= 0))[0] + 1

    # Only keep peaks after time >= 15
    valid_peaks = peaks[time[peaks] >= 15]
    all_peaks.append(valid_peaks)

    for p in valid_peaks:
        timestamp = time[p]
        amplitude = signal[p]
        peak_records.append((timestamp, i, p, amplitude))  # (time, sensor_id, index, amp)

# === Step 3: Cluster peaks ===
ref_idx = 0  # reference sensor
NUM_SENSORS = signals.shape[1]
fs = 1 / np.mean(np.diff(time))

# Build peak features (t, amp, TDOAs)
peak_features = []
for (t_i, sensor_id, p_idx, amp_i) in peak_records:
    feature = [t_i, amp_i]
    peak_features.append(feature)

features = np.array(peak_features)
scaled_features = StandardScaler().fit_transform(features)

# Cluster with GMM
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
labels = gmm.fit_predict(scaled_features)
probabilities = gmm.predict_proba(scaled_features)
max_probs = np.max(probabilities, axis=1)

# Flag low-confidence peaks
overlapping_peaks = max_probs < 0.95  # adjust threshold if needed

# Plot signals with clustered peaks
cluster_colors = ['red', 'blue']
overlap_color = 'purple'

plt.figure(figsize=(12, 2.5 * NUM_SENSORS))
for sensor_id in range(NUM_SENSORS):
    plt.subplot(NUM_SENSORS, 1, sensor_id + 1)
    plt.plot(time, signals[:, sensor_id], color='black', alpha=0.6, label=f"Sensor {sensor_id}")

    for i, (t_i, sid, p_idx, amp_i) in enumerate(peak_records):
        if sid == sensor_id:
            if overlapping_peaks[i]:
                plt.plot(t_i, signals[p_idx, sid], 'o', color=overlap_color, markersize=5)
            else:
                cluster = labels[i]
                plt.plot(t_i, signals[p_idx, sid], 'o', color=cluster_colors[cluster], markersize=5)

    plt.xlabel("Time (s)")
    plt.ylabel("Voltage")
    plt.title(f"Sensor {sensor_id}")
    plt.grid(True)

plt.tight_layout()
plt.suptitle("TDOA-Based Peak Clustering", y=1.02, fontsize=16)
plt.show()

# === Step 4: Group clustered peaks by label and sensor ===
cluster_peaks = {0: {}, 1: {}}

for (t_i, sensor_id, peak_idx, amp_i), label in zip(peak_records, labels):
    if sensor_id not in cluster_peaks[label]:
        cluster_peaks[label][sensor_id] = peak_idx
    else:
        # Use the earlier peak if multiple from same sensor assigned to same cluster
        if t_i < time[cluster_peaks[label][sensor_id]]:
            cluster_peaks[label][sensor_id] = peak_idx


# === Step 5: localization function ===
def grid_search_localization(sensor_coords, tdoas, ref_index=0, sound_speed=162900, grid_size=100):
    xmin, ymin = [2,2]
    xmax, ymax = [20,12]

    grid_x, grid_y = np.meshgrid(
        np.linspace(xmin, xmax, grid_size),
        np.linspace(ymin, ymax, grid_size)
    )

    error_grid = np.zeros_like(grid_x)
    ref_pos = sensor_coords[ref_index]
    ref_dist = np.sqrt((grid_x - ref_pos[0]) ** 2 + (grid_y - ref_pos[1]) ** 2)

    for i, (x_s, y_s) in enumerate(sensor_coords):
        if i == ref_index:
            continue
        dist = np.sqrt((grid_x - x_s) ** 2 + (grid_y - y_s) ** 2)
        expected_tdoa = (dist - ref_dist) / sound_speed
        error_grid += (tdoas[i] - expected_tdoa) ** 2

    min_idx = np.unravel_index(np.argmin(error_grid), error_grid.shape)
    est_x = grid_x[min_idx]
    est_y = grid_y[min_idx]
    return (est_x, est_y), error_grid, grid_x, grid_y


cluster_sources = {}

for cluster_id in cluster_peaks:
    peak_dict = cluster_peaks[cluster_id]

    # Only proceed if at least 3 sensors contributed peaks
    if len(peak_dict) < 3:
        print(f"Cluster {cluster_id}: Not enough peaks for localization.")
        continue

    # Get peak times for each sensor in the cluster
    tdoas = np.zeros(NUM_SENSORS)
    ref_time = time[peak_dict[ref_idx]]

    for i in range(NUM_SENSORS):
        if i in peak_dict:
            tdoas[i] = time[peak_dict[i]] - ref_time
        else:
            tdoas[i] = np.nan  # No peak from this sensor

    # Replace NaNs with 0 or interpolate
    nan_mask = np.isnan(tdoas)
    if np.sum(~nan_mask) < 3:
        print(f"Cluster {cluster_id}: Not enough valid TDOAs.")
        continue

    # Optional: zero-fill NaNs (or interpolate if needed)
    tdoas[nan_mask] = 0

    # Run grid search localization
    est_pos, error_map, grid_x, grid_y = grid_search_localization(sensor_coords, tdoas, ref_index=ref_idx)

    cluster_sources[cluster_id] = (est_pos, error_map)
    print(f"Estimated location for cluster {cluster_id}: {est_pos}")

# === Step 7: Separate signals by speaker cluster and analyze ===
speaker_signals = {0: np.zeros_like(signals), 1: np.zeros_like(signals)}
window_size = 15  # samples before and after peak

for (t, sensor_idx, peak_idx, amp), label in zip(peak_records, labels):
    start = max(0, peak_idx - window_size)
    end = min(len(time), peak_idx + window_size)
    speaker_signals[label][start:end, sensor_idx] += signals[start:end, sensor_idx]

# Sampling frequency
fs = 1 / np.mean(np.diff(time))
true_freqs = [2.36, 2.867]  # already sorted: low → high
estimated_freqs = [0, 0]

# === Step 8: Remap cluster labels so Cluster 0 = lower freq, Cluster 1 = higher freq ===
# === Step 1: Check frequency spread in each cluster ===
cluster_peak_freqs = {}
cluster_needs_reassign = False

for label in [0, 1]:
    avg_signal = np.mean(speaker_signals[label], axis=1)
    fft_magnitude = np.abs(np.fft.rfft(avg_signal))
    frequencies = np.fft.rfftfreq(len(avg_signal), d=1 / fs)

    peak_indices, _ = find_peaks(fft_magnitude, height=0.3 * np.max(fft_magnitude))
    peak_freqs = frequencies[peak_indices]
    cluster_peak_freqs[label] = peak_freqs

    if len(peak_freqs) >= 2:
        min_peak = np.min(peak_freqs)
        max_peak = np.max(peak_freqs)
        spread = max_peak - min_peak
        print(f"Cluster {label}: min = {min_peak:.2f} Hz, max = {max_peak:.2f} Hz, spread = {spread:.2f} Hz")

        if spread > 0.35:
            print(f"⚠️ Cluster {label} exceeds 0.35 Hz spread → will reassign PEAKS.")
            cluster_needs_reassign = True

# === Step 2: Reassign peaks based on proximity to min vs. max freq ===
if cluster_needs_reassign:
    print("🔁 Reassigning PEAKS based on proximity to min/max peak frequency...")

    # Determine global min/max peak frequencies from both clusters
    all_peaks = np.concatenate([cluster_peak_freqs[0], cluster_peak_freqs[1]])
    global_min_peak = np.min(all_peaks)
    global_max_peak = np.max(all_peaks)

    # Reassign each peak
    reassigned_labels = []
    peak_freqs = []

    for (t, sensor_id, peak_idx, amp) in peak_records:
        signal = signals[:, sensor_id]
        segment = signal[max(0, peak_idx - window_size):min(len(signal), peak_idx + window_size)]
        fft_mag = np.abs(np.fft.rfft(segment))
        freqs = np.fft.rfftfreq(len(segment), d=1/fs)
        peak_idx_seg, _ = find_peaks(fft_mag, height=0.3 * np.max(fft_mag))
        segment_peaks = freqs[peak_idx_seg]

        if len(segment_peaks) == 0:
            dominant_freq = 0  # fallback
        else:
            dominant_freq = np.mean(segment_peaks)

        peak_freqs.append(dominant_freq)
        # Assign label based on proximity to min or max
        if abs(dominant_freq - global_min_peak) < abs(dominant_freq - global_max_peak):
            reassigned_labels.append(0)
        else:
            reassigned_labels.append(1)

    labels = reassigned_labels
    print("✅ Reassignment complete.")

else:
    print("✅ No reassignment needed.")

from scipy.signal import find_peaks

# === Step 3: Remap labels so Cluster 0 = lower freq ===
print("=== Final Remap: Cluster 0 = lower frequency ===")

# Estimate median frequency per cluster based on peak segment FFTs
cluster_freqs = {0: [], 1: []}

for (t, sensor_id, peak_idx, amp), label in zip(peak_records, labels):
    segment = signals[max(0, peak_idx - window_size):min(len(signals), peak_idx + window_size), sensor_id]
    fft_mag = np.abs(np.fft.rfft(segment))
    freqs = np.fft.rfftfreq(len(segment), d=1/fs)
    peak_indices, _ = find_peaks(fft_mag, height=0.3 * np.max(fft_mag))
    if len(peak_indices) > 0:
        dominant_freq = np.mean(freqs[peak_indices])
        cluster_freqs[label].append(dominant_freq)

median_freqs = [np.median(cluster_freqs[0]), np.median(cluster_freqs[1])]
lower_label = int(np.argmin(median_freqs))
higher_label = 1 - lower_label
label_map = {lower_label: 0, higher_label: 1}
print(f"Remapping labels: {label_map} (lower freq → Cluster 0, higher freq → Cluster 1)")

# Apply remapping
remapped_labels = np.array([label_map[l] for l in labels])
speaker_signals = {0: np.zeros_like(signals), 1: np.zeros_like(signals)}
cluster_peaks = {0: {}, 1: {}}

for (t, sensor_id, peak_idx, amp), original_label in zip(peak_records, labels):
    new_label = label_map[original_label]
    start = max(0, peak_idx - window_size)
    end = min(len(time), peak_idx + window_size)
    speaker_signals[new_label][start:end, sensor_id] += signals[start:end, sensor_id]
    if sensor_id not in cluster_peaks[new_label] or t < time[cluster_peaks[new_label][sensor_id]]:
        cluster_peaks[new_label][sensor_id] = peak_idx

# === Step 9: Visualizations ===

# --- Plot spectrum and estimated frequency per cluster ---
estimated_freqs = {}

fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for label in [0, 1]:
    avg_signal = np.mean(speaker_signals[label], axis=1)
    fft_magnitude = np.abs(np.fft.rfft(avg_signal))
    frequencies = np.fft.rfftfreq(len(avg_signal), d=1 / fs)

    peak_indices, _ = find_peaks(fft_magnitude, height=0.6 * np.max(fft_magnitude))
    peak_freqs = frequencies[peak_indices]
    peak_mags = fft_magnitude[peak_indices]

    estimated_freq = np.median(peak_freqs) if len(peak_freqs) > 0 else 0
    estimated_freqs[label] = estimated_freq

    ax = axs[label]
    ax.plot(frequencies, fft_magnitude, label='Spectrum')
    ax.plot(peak_freqs, peak_mags, 'o', label='Peaks')

    for f, m in zip(peak_freqs, peak_mags):
        ax.annotate(f'{f:.2f} Hz', xy=(f, m), xytext=(5, 5),
                    textcoords='offset points', fontsize=8)

    ax.set_title(f"Cluster {label} Frequency Spectrum")
    ax.set_xlabel("Frequency (Hz)")
    ax.grid(True)
    ax.legend()

    # Frequency error printout
    percent_error = abs(estimated_freq - true_freqs[label]) / true_freqs[label] * 100
    print(f"Cluster {label}:")
    print(f"  Estimated Frequency: {estimated_freq:.3f} Hz")
    print(f"  True Frequency: {true_freqs[label]:.2f} Hz")
    print(f"  Percent Error: {percent_error:.2f}%\n")

axs[0].set_ylabel("Magnitude")
plt.tight_layout()
plt.show()

# --- Plot voltage signals over time ---
plt.figure(figsize=(12, 6))
for label in [0, 1]:
    avg_signal = np.mean(speaker_signals[label], axis=1)
    plt.plot(time, avg_signal, label=f'Cluster {label} Avg Voltage')

plt.title("Remapped Voltage Signals Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (V)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()