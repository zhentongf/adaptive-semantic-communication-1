import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd

# Load SNR values (from your code's output or saved log)
snr_values = np.array([0.50, 0.60, 1.74, 4.46, 5.50, 8.44, 12.79, 13.53, 14.73, 17.84])

# Load Accuracy (replace with your CSV paths)
acc_files_1 = [
    "acc_semantic_combining_0.70_snr_0.50.csv",
    "acc_semantic_combining_0.70_snr_0.60.csv",
    "acc_semantic_combining_0.70_snr_1.74.csv",
    "acc_semantic_combining_0.70_snr_4.46.csv",
    "acc_semantic_combining_0.70_snr_5.50.csv",
    "acc_semantic_combining_0.70_snr_8.44.csv",
]
acc_files_2 = [
    "acc_semantic_combining_0.70_snr_12.79.csv",
    "acc_semantic_combining_0.70_snr_13.53.csv",
    "acc_semantic_combining_0.70_snr_14.73.csv",
    "acc_semantic_combining_0.70_snr_17.84.csv",
]

accuracy_1 = np.array([pd.read_csv("results/MLP_sem_CIFAR-02/"+f).iloc[-1, 0] for f in acc_files_1])
accuracy_2 = np.array([pd.read_csv("results/MLP_sem_CIFAR/"+f).iloc[-1, 0] for f in acc_files_2])
acc_raw = np.concatenate((accuracy_1, accuracy_2))

acc_adaptive = np.array([pd.read_csv("results/MLP_sem_CIFAR/"+f).iloc[-1, 0] for f in acc_files_1 + acc_files_2])

accuracy_1 = np.array([pd.read_csv("results/MLP_sem_CIFAR/"+f).iloc[5, 0] for f in acc_files_1])
accuracy_2 = np.array([pd.read_csv("results/MLP_sem_CIFAR-02/"+f).iloc[5, 0] for f in acc_files_2])
acc_semantic = np.concatenate((accuracy_1, accuracy_2))




# ----------------------------
# Step 2: Plot the Figure
# ----------------------------
plt.style.use('seaborn-v0_8-ticks')  # Elegant style for academic papers
fig, ax1 = plt.subplots(figsize=(8, 5))  # Figure size: 8x5 inches

# Plot Accuracy (left Y-axis)
color = 'tab:blue'
ax1.set_xlabel('SNR (dB)', fontsize=12)
ax1.set_ylabel('Accuracy', color=color, fontsize=12)
ax1.plot(snr_values, acc_raw, marker='o', linestyle='-', color='tab:blue', linewidth=2, markersize=6, label='acc_raw')
ax1.plot(snr_values, acc_adaptive, marker='^', linestyle='-', color='tab:green', linewidth=2, markersize=6, label='acc_adaptive')
ax1.plot(snr_values, acc_semantic, marker='s', linestyle='-', color='tab:orange', linewidth=2, markersize=6, label='acc_semantic')

ax1.tick_params(axis='y', labelcolor=color)
# ax1.set_ylim(0.4, 1.0)  # Accuracy range: 0.4-1.0
ax1.xaxis.set_major_locator(MaxNLocator(integer=False))  # Ensure SNR ticks are readable
ax1.grid(alpha=0.3)


# Add title and legend
plt.title('Comparison of Raw, Adaptive and Semantic for CIFAR-10 (Compression Rate = 0.7)', fontsize=14, pad=15)
lines, labels = ax1.get_legend_handles_labels()
ax1.legend(lines, labels, loc='lower right', fontsize=10)

# Adjust layout and save
plt.tight_layout()  # Avoid label cutoff
plt.savefig('Comparison of Raw, Adaptive and Semantic.pdf', dpi=300, bbox_inches='tight')  # Save as PDF for high-quality printing
plt.show()