import glob
import numpy as np
import matplotlib.pyplot as plt
import re

def process_files_with_list(file_pattern="*.txt", smoothing=1, time_range=None):
    # Function for smoothing
    def smooth(data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode="valid")

    # Dictionary to store data
    data_dict = {}
    change_points = set()

    prev_condition = None  # Initialize prev_condition here

    # Iterate over each file matching the pattern
    for filename in glob.glob(file_pattern):
        with open(filename, "r") as file:
            for line in file:
                # ... (same code as before)

    # Create subplots for the four lines outside the condition loop
    fig, axes = plt.subplots(2, 2)

    # Process and print results, and plot
    for condition, values in data_dict.items():
        print(f"Results for condition: {condition}")
        for i in range(4):
            ax = axes[i // 2, i % 2]  # Select the subplot
            averages = values[i]

            # Apply smoothing if applicable
            if smoothing > 1:
                averages = smooth(averages, smoothing)

            # Plot the results for this element with label as the condition
            ax.plot(averages, label=condition)
            ax.set_title(f"Element {i}")
            for change_point in change_points:
                ax.axvline(x=change_point, color="grey", linestyle="--")
            ax.legend()  # Add legend to show condition labels

    # Add labels to the subplots
    fig.text(0.5, 0.04, "Index", ha="center")
    fig.text(0.04, 0.5, "Value", va="center", rotation="vertical")
    plt.suptitle("Results for all conditions")
    plt.show()

# Example usage
process_files_with_list(file_pattern="replace*.txt", smoothing=1)
