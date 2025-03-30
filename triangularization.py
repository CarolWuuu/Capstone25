import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# === Step 2: Detect two peaks per sensor ===
peak_times = np.full((NUM_SENSORS, 2), np.nan)
all_peaks = []
first_two_peaks = []

for i in range(NUM_SENSORS):
    signal = signals[:, i]
    gradient_signal = np.diff(signal)
    peaks = np.where((gradient_signal[:-1] > 0) & (gradient_signal[1:] <= 0))[0] + 1

    # Filter peaks after 15 seconds
    peaks = peaks[time[peaks] >= 15.0]
    all_peaks.append(peaks)
    peak_values = signal[peaks]

    if len(peaks) >= 2:
        first_two_peaks.append(peaks[:2])
        for j in range(2):
            peak_times[i, j] = time[peaks[j]]
    else:
        first_two_peaks.append(np.array([]))

# === Plot first 2 detected peaks from different sensors ===
fig, axs = plt.subplots(NUM_SENSORS, 1, figsize=(12, 10), sharex=True)
for i in range(NUM_SENSORS):
    signal = signals[:, i]
    axs[i].plot(time, signal, label=f'Sensor {i+1} Signal')
    if len(first_two_peaks[i]) > 0:
        for j, peak_idx in enumerate(first_two_peaks[i]):
            t = time[peak_idx]
            axs[i].axvline(t, color='red', linestyle='--')
            axs[i].plot(t, signal[peak_idx], 'rx')
            axs[i].text(t + 0.01, signal[peak_idx], f"{t:.2f}s", color='red', fontsize=8)
    axs[i].set_ylabel("Amplitude")
    axs[i].legend()
    axs[i].grid(True)
axs[-1].set_xlabel("Time (s)")
plt.suptitle("First Two Detected Peaks for All Sensors")
plt.tight_layout()
plt.show()

# === Step 3: Trilateration Function Using Circle Intersections ===
def trilaterate_intersections(sensor_coords, distances, ax=None):
    from itertools import combinations
    centers = sensor_coords
    radii = distances
    points = []
    for i, j in combinations(range(len(centers)), 2): # Run every possible pair of sensors
        x0, y0, r0 = *centers[i], radii[i]
        x1, y1, r1 = *centers[j], radii[j]
        dx, dy = x1 - x0, y1 - y0
        d = np.hypot(dx, dy) # distance from two centers
        if d > r0 + r1 or d < abs(r0 - r1): # if do not intersect skip the pair
            continue  # no solution
        a = (r0**2 - r1**2 + d**2) / (2 * d) # d from center1 to midpoint bw 2 intersections
        h = np.sqrt(r0**2 - a**2) # d from midpoint to intersection
        xm = x0 + a * dx / d
        ym = y0 + a * dy / d
        xs1 = xm + h * dy / d
        ys1 = ym - h * dx / d
        xs2 = xm - h * dy / d
        ys2 = ym + h * dx / d
        points.extend([(xs1, ys1), (xs2, ys2)]) # intersection points
        if ax is not None:
            ax.plot([xs1, xs2], [ys1, ys2], 'k+', label='Intersection' if len(points) == 2 else "")
    if points:
        points = np.array(points)
        return np.mean(points, axis=0)
    else:
        raise ValueError("No intersection points found")

# === Step 4: Estimate Positions for Two Sources ===
estimated_positions = []
true_positions = np.array([[7.0, 16.0], [7.0, 6.0]])  # in cm

fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

for k in range(2):
    ax = axs[k]
    if np.isnan(peak_times[:, k]).any():
        print(f"Some peak times for Source {k+1} are missing. Skipping.")
        ax.set_title(f"Source {k+1}: Incomplete Data")
        continue

    time_diffs = peak_times[:, k] - np.min(peak_times[:, k])  # seconds
    distances = SPEED_OF_SOUND * time_diffs  # cm

    try:
        x, y = trilaterate_intersections(sensor_coords, distances, ax=ax)
        estimated_positions.append([x, y])
        print(f"Estimated Source {k+1} Position: x = {x:.2f} cm, y = {y:.2f} cm")
    except Exception as e:
        print(f"Trilateration failed for Source {k+1}: {e}")
        continue

    ax.set_title(f"Source {k+1} Distance Circles & Intersections")
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_aspect('equal')
    ax.grid(True)

    for i, (sx, sy) in enumerate(sensor_coords):
        circle = plt.Circle((sx, sy), distances[i], color='blue', alpha=0.2)
        ax.add_patch(circle)
        ax.plot(sx, sy, 'bs')
        ax.text(sx + 0.5, sy, f"S{i+1}", fontsize=8)

    ax.plot(*true_positions[k], 'gx', markersize=10, label='True Source')
    ax.plot(x, y, 'ro', markersize=8, label='Estimated Source')
    ax.legend()

plt.tight_layout()
plt.show()

estimated_positions = np.array(estimated_positions)

# === Step 5: Summary Plot ===
plt.figure()
plt.scatter(sensor_coords[:, 0], sensor_coords[:, 1], c='blue', marker='s', label='Sensors')
plt.scatter(true_positions[:, 0], true_positions[:, 1], c='green', marker='x', label='True Sources')
plt.scatter(estimated_positions[:, 0], estimated_positions[:, 1], c='red', marker='o', label='Estimated Sources')
plt.title("Source Localization via Circle Intersections (Trilateration)")
plt.xlabel("X (cm)")
plt.ylabel("Y (cm)")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

# === Step 6: Accuracy Measurement ===
if estimated_positions.shape == true_positions.shape:
    errors = np.linalg.norm(estimated_positions - true_positions, axis=1)
    for i, error in enumerate(errors):
        print(f"Source {i+1} Localization Error: {error:.2f} cm")
    mean_error = np.mean(errors)
    print(f"Mean Localization Error: {mean_error:.2f} cm")
else:
    print("Mismatch in number of estimated and true positions. Cannot compute error.")



