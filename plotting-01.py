import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd

# ----------------------------
# Step 1: Prepare Data (Simulated/Replace with Actual Data)
# ----------------------------
# According to your code logic: 10 random SNR values (0-20 dB) with compression rate 0.1
# Simulate SNR values (matching the "random.seed(42)" style in your code)
# np.random.seed(42)
# snr_values = np.sort(np.random.uniform(0, 20, 10))  # 10 SNR values (sorted for better visualization)

# # Simulate Accuracy and PSNR (based on communication principles: higher SNR → better performance)
# # Accuracy: 0.5-0.95 (increases with SNR)
# accuracy = 0.5 + 0.45 * (1 - np.exp(-0.15 * snr_values))  # Sigmoid-like growth

# # PSNR: 15-30 dB (increases with SNR)
# psnr = 15 + 15 * (1 - np.exp(-0.1 * snr_values))  # Sigmoid-like growth

# If you have actual data, replace the above with:
# Example of loading actual data (uncomment and modify paths)

# Load SNR values (from your code's output or saved log)
snr_values = np.array([0.50, 0.60, 1.74, 4.46, 5.50, 8.44])  # e.g., [0.50, 0.60, 1.74, 4.46, 5.50, 8.44, ...]

# Load Accuracy (replace with your CSV paths)
acc_files = [
    "acc_semantic_combining_0.100000_snr_0.50.csv",
    "acc_semantic_combining_0.100000_snr_0.60.csv",
    "acc_semantic_combining_0.100000_snr_1.74.csv",
    "acc_semantic_combining_0.100000_snr_4.46.csv",
    "acc_semantic_combining_0.100000_snr_5.50.csv",
    "acc_semantic_combining_0.100000_snr_8.44.csv",
]
accuracy = np.array([pd.read_csv("results/MLP_sem_MNIST/"+f).iloc[-1, 0] for f in acc_files])  # Use last epoch's accuracy

# Load PSNR (replace with your CSV paths)
psnr_files = [
    "psnr_semantic_combining_0.100000_snr_0.50.csv",
    "psnr_semantic_combining_0.100000_snr_0.60.csv",
    "psnr_semantic_combining_0.100000_snr_1.74.csv",
    "psnr_semantic_combining_0.100000_snr_4.46.csv",
    "psnr_semantic_combining_0.100000_snr_5.50.csv",
    "psnr_semantic_combining_0.100000_snr_8.44.csv",
]
psnr = np.array([pd.read_csv("results/MLP_sem_MNIST/"+f).iloc[-1, 0] for f in psnr_files])  # Use last epoch's PSNR



# ----------------------------
# Step 2: Plot the Figure
# ----------------------------
plt.style.use('seaborn-v0_8-ticks')  # Elegant style for academic papers
fig, ax1 = plt.subplots(figsize=(8, 5))  # Figure size: 8x5 inches

# Plot Accuracy (left Y-axis)
color = 'tab:blue'
ax1.set_xlabel('SNR (dB)', fontsize=12)
ax1.set_ylabel('Accuracy', color=color, fontsize=12)
ax1.plot(snr_values, accuracy, marker='o', linestyle='-', color=color, linewidth=2, markersize=6, label='Accuracy')
ax1.tick_params(axis='y', labelcolor=color)
# ax1.set_ylim(0.4, 1.0)  # Accuracy range: 0.4-1.0
ax1.xaxis.set_major_locator(MaxNLocator(integer=False))  # Ensure SNR ticks are readable
ax1.grid(alpha=0.3)

# Create a second Y-axis for PSNR
ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('PSNR (dB)', color=color, fontsize=12)  
ax2.plot(snr_values, psnr, marker='s', linestyle='--', color=color, linewidth=2, markersize=6, label='PSNR')
ax2.tick_params(axis='y', labelcolor=color)
# ax2.set_ylim(10, 35)  # PSNR range: 10-35 dB

# Add title and legend
plt.title('Accuracy and PSNR vs. SNR for MNIST (Compression Rate = 0.1)', fontsize=14, pad=15)
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='lower right', fontsize=10)

# Adjust layout and save
plt.tight_layout()  # Avoid label cutoff
plt.savefig('SNR_Accuracy_PSNR.pdf', dpi=300, bbox_inches='tight')  # Save as PDF for high-quality printing
plt.show()