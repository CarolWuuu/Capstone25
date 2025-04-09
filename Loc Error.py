import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv("Est Loc.csv")

# Define true speaker locations and colors
true_locations = {
    'A': {'coords': [7, 16], 'color': 'red', 'label': 'Speaker 1 (A)'},
    'B': {'coords': [7, 6], 'color': 'blue', 'label': 'Speaker 2 (B)'}
}
# Create updated figure with corrected hydrogel bounds and filled error circles
fig, ax = plt.subplots(figsize=(8, 10))

# Updated hydrogel boundary
ax.set_xlim(0, 14)
ax.set_ylim(0, 22)
ax.set_aspect('equal')
ax.set_title("True and Estimated Speaker Locations with Filled Error Circles")
ax.set_xlabel("X (cm)")
ax.set_ylabel("Y (cm)")

# Plot estimates and error regions
for speaker, info in true_locations.items():
    true_x, true_y = info['coords']
    color = info['color']

    # Filter estimates
    speaker_df = df[df['Speaker'] == speaker]
    est_x = speaker_df['Est. X (cm)']
    est_y = speaker_df['Est. Y (cm)']

    # Plot estimated points
    ax.scatter(est_x, est_y, color=color, alpha=0.6, label=f"{info['label']} Estimates")

    # Plot true location
    ax.plot(true_x, true_y, marker='*', color=color, markersize=15, label=f"{info['label']} True Location")

    # Calculate mean error radius
    error_radius = np.mean(np.sqrt((est_x - true_x) ** 2 + (est_y - true_y) ** 2))

    # Draw filled error circle
    circle = plt.Circle((true_x, true_y), error_radius, color=color, alpha=0.2, label=f"{info['label']} Error Range")
    ax.add_patch(circle)

# Add legend and grid
ax.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("location_estimation_error.png", dpi=300, bbox_inches='tight')
plt.show()
