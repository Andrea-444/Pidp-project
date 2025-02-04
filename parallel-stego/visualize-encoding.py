import matplotlib.pyplot as plt

image_pairs = [
    "490x750 - 417x626", "608x405 - 1351x1080", "1280x856 - 2560x1711",
    "2160x2700 - 6048x8064", "4000x2667 - 6000x4000", "4972x7458 - 2667x4000"
]

# seq
sequential_times = [2.16, 2.11, 7.79, 39.67, 69.38, 71.27]

# parallel
parallel_times = [2.23, 2.22, 2.44, 3.89, 3.76, 4.52]

plt.figure(figsize=(10, 6))

plt.plot(image_pairs, sequential_times, marker='o', label='Sequential Approach', color='red', linestyle='-', linewidth=2)

plt.plot(image_pairs, parallel_times, marker='s', label='Parallel Approach', color='blue', linestyle='--', linewidth=2)

plt.xlabel("Image Pair (Cover x Secret)", fontsize=12)
plt.ylabel("Execution Time (Seconds)", fontsize=12)
plt.title("Sequential vs Parallel Execution Time for Encoding", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
#
# import matplotlib.pyplot as plt
#
# # Data: Replace with your actual execution times
# image_pairs = [
#     "490x750 - 417x626", "608x405 - 1351x1080", "1280x856 - 2560x1711",
#     "2160x2700 - 6048x8064", "4000x2667 - 6000x4000", "4972x7458 - 2667x4000"
# ]
#
# # Sequential execution times (in seconds) for each image pair
# sequential_times = [2.16, 2.11, 7.79, 39.67, 69.38, 71.27]
#
# # Parallel execution times (in seconds) for each image pair
# parallel_times = [2.23, 2.22, 2.44, 3.89, 3.76, 4.52]
#
# # Calculate speedup (Sequential / Parallel)
# speedup = [seq / par for seq, par in zip(sequential_times, parallel_times)]
#
# # Plot
# plt.figure(figsize=(10, 6))
#
# # Line plot for speedup
# plt.plot(image_pairs, speedup, marker='o', label='Speedup (Seq / Parallel)', color='blue', linestyle='-', linewidth=2)
#
# # Formatting the plot
# plt.xlabel("Image Pair (Cover x Secret)", fontsize=12)
# plt.ylabel("Speedup (Sequential / Parallel)", fontsize=12)
# plt.title("Speedup of Parallel Execution vs Sequential Execution", fontsize=14)
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.legend()
#
# # Rotate x-axis labels for better readability
# plt.xticks(rotation=45)
#
# # Show plot
# plt.tight_layout()
# plt.show()
