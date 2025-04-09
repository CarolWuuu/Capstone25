# Load the newly uploaded CSV file
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

avg_error_df = pd.read_csv("avg FHR percent error.csv")

# Preview the structure
avg_error_df.head()

# Create pivot tables for each speaker
pivot_A = avg_error_df[avg_error_df['Speaker'] == 'A'].pivot(
    index='True FHR A (bpm)', columns='True FHR B (bpm)', values='Average % Error')
pivot_B = avg_error_df[avg_error_df['Speaker'] == 'B'].pivot(
    index='True FHR A (bpm)', columns='True FHR B (bpm)', values='Average % Error')

# Sort indices and columns for consistent heatmap orientation
pivot_A = pivot_A.sort_index().sort_index(axis=1)
pivot_B = pivot_B.sort_index().sort_index(axis=1)

# Plotting
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

sns.heatmap(pivot_A, annot=True, fmt=".2f", cmap='viridis', ax=axs[0], cbar_kws={'label': '% Error'})
axs[0].set_title("A\nPercent Error for Fetus A (Speaker A)", fontsize=14)
axs[0].set_xlabel("Fetus B Heart Rate (bpm)")
axs[0].set_ylabel("Fetus A Heart Rate (bpm)")

sns.heatmap(pivot_B, annot=True, fmt=".2f", cmap='viridis', ax=axs[1], cbar_kws={'label': '% Error'})
axs[1].set_title("B\nPercent Error for Fetus B (Speaker B)", fontsize=14)
axs[1].set_xlabel("Fetus B Heart Rate (bpm)")
axs[1].set_ylabel("Fetus A Heart Rate (bpm)")

plt.tight_layout()
plt.savefig("avg_percent_error_heatmaps.png", dpi=300, bbox_inches='tight')
plt.show()
