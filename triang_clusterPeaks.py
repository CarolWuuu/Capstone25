import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# === Setup ===
FILENAME = 'filtered_piezo_voltage_log.csv'
SPEED_OF_SOUND = 162900  # cm/s in hydrogel

# Sensor coordinates (centimeters)
sensor_coords = np.array([
    [9.0, 8.5],
    [11.7, 15.5],
    [3.5, 2.5],
    [2.5, 11.5],
    [5.0, 18.5],
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
    gradient_signal = np.diff(signal)
    peaks = np.where((gradient_signal[:-1] > 0) & (gradient_signal[1:] <= 0))[0] + 1
    peaks = peaks[time[peaks] >= 15.0]
    all_peaks.append(peaks)
    for p in peaks:
        peak_records.append((time[p], i, p))  # (timestamp, sensor_idx, peak_idx)

# === Step 3: Cluster peaks based on arrival time ===
peak_times = np.array([[t] for t, _, _ in peak_records])
if len(peak_times) >= 2:
    kmeans = KMeans(n_clusters=2, random_state=0).fit(peak_times)
    labels = kmeans.labels_
else:
    labels = np.zeros(len(peak_records), dtype=int)

# === Step 4: Group clustered peaks by label and sensor ===
cluster_peaks = {0: {}, 1: {}}
for (t, sensor_idx, peak_idx), label in zip(peak_records, labels):
    if sensor_idx not in cluster_peaks[label]:
        cluster_peaks[label][sensor_idx] = peak_idx
    else:
        if time[peak_idx] < time[cluster_peaks[label][sensor_idx]]:
            cluster_peaks[label][sensor_idx] = peak_idx  # use earlier if multiple

# === Step 5: Trilateration function ===
def trilaterate_intersections(sensor_coords, distances):
    from itertools import combinations
    centers = sensor_coords
    radii = distances
    points = []
    for i, j in combinations(range(len(centers)), 2):
        x0, y0, r0 = *centers[i], radii[i]
        x1, y1, r1 = *centers[j], radii[j]
        dx, dy = x1 - x0, y1 - y0
        d = np.hypot(dx, dy)
        if d > r0 + r1 or d < abs(r0 - r1):
            continue
        a = (r0**2 - r1**2 + d**2) / (2 * d)
        h = np.sqrt(r0**2 - a**2)
        xm = x0 + a * dx / d
        ym = y0 + a * dy / d
        xs1 = xm + h * dy / d
        ys1 = ym - h * dx / d
        xs2 = xm - h * dy / d
        ys2 = ym + h * dx / d
        points.extend([(xs1, ys1), (xs2, ys2)])
    if points:
        points = np.array(points)
        return np.mean(points, axis=0), points
    else:
        raise ValueError("No intersection points found")

# === Step 6: Estimate location for each cluster and plot ===
estimated_positions = []
all_intersection_points = []
colors = ['red', 'green']

# Clustered peaks per sensor
fig, axs = plt.subplots(NUM_SENSORS, 1, figsize=(12, 10), sharex=True)
for i in range(NUM_SENSORS):
    axs[i].plot(time, signals[:, i], label=f'Sensor {i+1} Signal')
    for label, color in enumerate(colors):
        if i in cluster_peaks[label]:
            peak_idx = cluster_peaks[label][i]
            t = time[peak_idx]
            axs[i].plot(t, signals[peak_idx, i], 'x', color=color)
            axs[i].axvline(t, linestyle='--', color=color, alpha=0.3)
            axs[i].text(t + 0.01, signals[peak_idx, i], f"{t:.2f}s", color=color, fontsize=8)
    axs[i].set_ylabel("Amplitude")
    axs[i].legend()
    axs[i].grid(True)
axs[-1].set_xlabel("Time (s)")
plt.suptitle("Clustered Peak Assignments by Sensor")
plt.tight_layout()
plt.show()

# Plot circle ranges and compute accuracy
true_positions = np.array([[7.0, 16.0], [7.0, 6.0]])
fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

for label in [0, 1]:
    ax = axs[label]
    if len(cluster_peaks[label]) < 3:
        ax.set_title(f"Cluster {label}: Incomplete")
        continue
    sensor_idxs = sorted(cluster_peaks[label].keys())
    peak_idxs = [cluster_peaks[label][i] for i in sensor_idxs]
    times = np.array([time[p] for p in peak_idxs])
    coords = np.array([sensor_coords[i] for i in sensor_idxs])
    t0 = np.min(times)
    distances = SPEED_OF_SOUND * (times - t0)
    (x, y), intersections = trilaterate_intersections(coords, distances)
    estimated_positions.append([x, y])
    all_intersection_points.append(intersections)

    for i, (cx, cy) in enumerate(coords):
        ax.add_patch(plt.Circle((cx, cy), distances[i], color=colors[label], alpha=0.2))
        ax.plot(cx, cy, 'bs')
        ax.text(cx + 0.5, cy, f"S{i+1}", fontsize=8)

    for (ix, iy) in intersections:
        ax.plot(ix, iy, 'k+', markersize=6)

    ax.plot(true_positions[label][0], true_positions[label][1], 'gx', markersize=10, label='True')
    ax.plot(x, y, 'o', color=colors[label], markersize=8, label='Estimated')
    ax.set_title(f"Cluster {label} Estimated Location")
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.grid(True)
    ax.set_aspect('equal')
    ax.legend()

plt.tight_layout()
plt.show()

# Summary plot and accuracy
estimated_positions = np.array(estimated_positions)
plt.figure()
plt.scatter(sensor_coords[:, 0], sensor_coords[:, 1], c='blue', marker='s', label='Sensors')
plt.scatter(true_positions[:, 0], true_positions[:, 1], c='green', marker='x', label='True Sources')
plt.scatter(estimated_positions[:, 0], estimated_positions[:, 1], c='red', marker='o', label='Estimated Sources')
plt.title("Summary: True vs Estimated Source Locations")
plt.xlabel("X (cm)")
plt.ylabel("Y (cm)")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

# Print estimated locations
for i, (x, y) in enumerate(estimated_positions):
    print(f"Estimated Source {i+1} Position: x = {x:.2f} cm, y = {y:.2f} cm")

# Accuracy
if estimated_positions.shape == true_positions.shape:
    errors = np.linalg.norm(estimated_positions - true_positions, axis=1)
    for i, err in enumerate(errors):
        print(f"Source {i+1} Error: {err:.2f} cm")
    print(f"Mean Localization Error: {np.mean(errors):.2f} cm")
else:
    print("Error: Position mismatch")

# === Step 7: Separate signals by speaker cluster and analyze ===
speaker_signals = {0: np.zeros_like(signals), 1: np.zeros_like(signals)}
window_size = 50  # samples before and after peak

for (t, sensor_idx, peak_idx), label in zip(peak_records, labels):
    start = max(0, peak_idx - window_size)
    end = min(len(time), peak_idx + window_size)
    speaker_signals[label][start:end, sensor_idx] += signals[start:end, sensor_idx]

# Frequency analysis for each speaker
fs = 1 / np.mean(np.diff(time))  # sampling frequency in Hz
frequencies = np.fft.rfftfreq(len(time), d=1/fs)

plt.figure(figsize=(12, 5))
for label in [0, 1]:
    # composite signal: averages across all sensors for each time point, producing a 1D signal
    avg_signal = np.mean(speaker_signals[label], axis=1)
    fft_magnitude = np.abs(np.fft.rfft(avg_signal))
    plt.plot(frequencies, fft_magnitude, label=f'Speaker {label+1}')

plt.title("Frequency Spectrum of Separated Speaker Signals")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot separated voltage over time for each speaker ===
plt.figure(figsize=(12, 6))
for label in [0, 1]:
    avg_signal = np.mean(speaker_signals[label], axis=1)
    plt.plot(time, avg_signal, label=f'Speaker {label+1}')

plt.title("Separated Voltage Signals Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (V)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
