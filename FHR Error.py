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

# Flip y-axis by sorting index in descending order
pivot_A = pivot_A.sort_index(ascending=False).sort_index(axis=1)
pivot_B = pivot_B.sort_index(ascending=False).sort_index(axis=1)

# Create custom colormap with grey for NaNs
cmap = sns.color_palette("viridis", as_cmap=True)
cmap.set_bad(color='gray')

# Plot
fig, axs = plt.subplots(2, 1, figsize=(5, 6))

sns.heatmap(pivot_A, annot=True, fmt=".2f", cmap=cmap, mask=pivot_A.isna(),
            ax=axs[0], cbar_kws={'label': '% Error'})
axs[0].set_title("A\nPercent Error for Fetus A (n=3)", fontsize=14)
axs[0].set_xlabel("Fetus B Heart Rate (bpm)")
axs[0].set_ylabel("Fetus A Heart Rate (bpm)")

sns.heatmap(pivot_B, annot=True, fmt=".2f", cmap=cmap, mask=pivot_B.isna(),
            ax=axs[1], cbar_kws={'label': '% Error'})
axs[1].set_title("B\nPercent Error for Fetus B (n=3)", fontsize=14)
axs[1].set_xlabel("Fetus B Heart Rate (bpm)")
axs[1].set_ylabel("Fetus A Heart Rate (bpm)")

plt.tight_layout()
plt.savefig("avg_percent_error_heatmaps.png", dpi=300, bbox_inches='tight')
plt.show()
