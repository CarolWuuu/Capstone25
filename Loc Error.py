import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load estimated data
df = pd.read_csv("All_Final_Estimated_Locations.csv")

# Sensor coordinates (cm)
sensor_coords = np.array([
    [9.0, 8.5],
    [11.7, 15.5],
    [3.5, 2.5],
    [2.5, 11.5],
    [5.0, 18.5],
])

# True speaker locations and colors
true_locations = {
    'A': {'coords': [7, 16], 'color': 'red', 'label': 'Speaker 1 (A)'},
    'B': {'coords': [7, 6], 'color': 'blue', 'label': 'Speaker 2 (B)'}
}

# Create figure
fig, ax = plt.subplots(figsize=(4, 5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 22)
ax.set_aspect('equal')
ax.set_title("True and Estimated Speaker Locations with Sensor Positions")
ax.set_xlabel("X (cm)")
ax.set_ylabel("Y (cm)")

# Plot sensor positions
sensor_x, sensor_y = sensor_coords[:, 0], sensor_coords[:, 1]
ax.scatter(sensor_x, sensor_y, marker='s', color='black', label='Sensor Locations')
for i, (x, y) in enumerate(sensor_coords):
    ax.text(x + 0.3, y, f"S{i+1}", fontsize=9, color='black')

# Plot estimates and error ranges
for speaker, info in true_locations.items():
    true_x, true_y = info['coords']
    color = info['color']

    speaker_df = df[df['Speaker'] == speaker]
    est_x = speaker_df['Est. X (cm)']
    est_y = speaker_df['Est. Y (cm)']

    ax.scatter(est_x, est_y, color=color, alpha=0.6, label=f"{info['label']} Estimates")
    ax.plot(true_x, true_y, marker='*', color=color, markersize=15)
    ax.text(true_x + 0.3, true_y + 0.3, info['label'], color=color, fontsize=10, fontweight='bold')

    error_radius = np.mean(np.sqrt((est_x - true_x) ** 2 + (est_y - true_y) ** 2))
    circle = plt.Circle((true_x, true_y), error_radius, color=color, alpha=0.2, label=f"{info['label']} Error Range")
    ax.add_patch(circle)

# Finalize and save
plt.grid(True)
plt.tight_layout()
plt.savefig("location_estimation_error.png", dpi=300, bbox_inches='tight')
plt.show()

