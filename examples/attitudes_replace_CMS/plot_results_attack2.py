import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def process_files_with_list(file_pattern="*.txt", smoothing=1, time_range=None):
    # Dictionary to store data
    data_dict = {}
    change_points = set()

    # Function for smoothing
    def smooth(data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode="valid")

    # Iterate over each file matching the pattern
    prev_condition = None
    for filename in glob.glob(file_pattern):
        with open(filename, "r") as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) < 8 or not parts[0].isdigit():
                    continue

                time_step = int(parts[0])
                condition = parts[-1]
                list_string = line.split("[")[-1].split("]")[0]
                four_values = [int(x) for x in list_string.split(",")]

                # Check if the time_step is within the specified range, if provided
                if time_range and (
                    time_step < time_range[0] or time_step > time_range[1]
                ):
                    continue

                # Check if the condition changes
                if prev_condition is not None and condition != prev_condition:
                    change_points.add(time_step)
                prev_condition = condition

                # Add data to dictionary
                if condition not in data_dict:
                    data_dict[condition] = {i: defaultdict(list) for i in range(4)}
                for i, val in enumerate(four_values):
                    data_dict[condition][i][time_step].append(val)

    # Process and print results, and plot
    fig, axes = plt.subplots(2, 2)  # Create subplots for the four lines
    for condition, values in data_dict.items():
        print(f"Results for condition: {condition}")
        for i in range(4):
            ax = axes[i // 2, i % 2]  # Select the subplot
            time_steps = sorted(values[i].keys())
            averages = []
            stderrs = []
            for t in time_steps:
                mean = np.mean(values[i][t])
                stderr = np.std(values[i][t]) / np.sqrt(len(values[i][t]))
                averages.append(mean)
                stderrs.append(stderr)

            # Apply smoothing if applicable
            if smoothing > 1:
                averages = smooth(averages, smoothing)
                stderrs = smooth(stderrs, smoothing)

            # Plot the results for this element
            ax.errorbar(
                time_steps[: len(averages)],
                averages,
                yerr=stderrs,
                label=f"{condition}",
            )
            ax.set_title(f"Element {i}")
            for change_point in change_points:
                ax.axvline(x=change_point, color="grey", linestyle="--")

        # Add labels to the subplots
        fig.text(0.5, 0.04, "Time Step", ha="center")
        fig.text(0.04, 0.5, "Value", va="center", rotation="vertical")
        plt.suptitle(f"Results")
    plt.legend()
    plt.show()


# Example usage
process_files_with_list(file_pattern="nn*.txt", smoothing=1)
