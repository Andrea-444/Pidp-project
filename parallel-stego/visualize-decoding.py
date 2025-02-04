import matplotlib.pyplot as plt
import numpy as np

cover_image_sizes = ["417x626", "608x405", "1280x856", "2160x2700", "4000x2667", "2667x4000"]
sequential_times = [1.09, 1.02, 4.67, 25.06, 43.93, 44.40]
parallel_times = [1.72, 1.75, 1.92, 2.21, 2.50, 2.17]

plt.figure(figsize=(10, 6))

# sequential times
plt.plot(cover_image_sizes, sequential_times, marker='o', label="Sequential Time", color='red', linestyle='-', linewidth=2)

# parallel times
plt.plot(cover_image_sizes, parallel_times, marker='o', label="Parallel Time", color='blue', linestyle='--', linewidth=2)

# # labels
# for i, (seq, par) in enumerate(zip(sequential_times, parallel_times)):
#     plt.text(i, seq + 1, f"{seq:.2f}s", color='red', ha='center', fontsize=10)  # Sequential time labels
#     plt.text(i, par - 1, f"{par:.2f}s", color='blue', ha='center', fontsize=10)  # Parallel time labels

plt.xlabel("Cover Image Size", fontsize=12)
plt.ylabel("Execution Time (Seconds)", fontsize=12)
plt.title("Sequential vs Parallel Times for Decoding", fontsize=14)
plt.xticks(rotation=45)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
