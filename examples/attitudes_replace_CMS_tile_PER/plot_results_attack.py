import glob
import numpy as np
import matplotlib.pyplot as plt
import re


def process_files_with_list(file_pattern="*.txt", smoothing=1, time_range=None):
    # Dictionary to store data
    data_dict = {}
    change_points = set()

    prev_condition = None  # Initialize prev_condition here

    # Function for smoothing
    def smooth(data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode="valid")

    # Iterate over each file matching the pattern
    for filename in glob.glob(file_pattern):
        with open(filename, "r") as file:
            for line in file:
                if "[" not in line or "]" not in line:  # Skip lines without brackets
                    continue

                parts = line.split()
                if (
                    len(parts) < 9 or not parts[0].isdigit()
                ):  # Skip lines without enough parts or without a numeric start
                    continue

                time_step = int(parts[0])

                # Use regular expressions to find the condition pattern
                condition_pattern = re.search(r"implicit_attitude\+EWA.*? ", line)
                if condition_pattern:
                    condition = condition_pattern.group().strip()
                else:
                    continue  # Skip lines without the pattern

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

                if condition not in data_dict:
                    data_dict[condition] = {i: [] for i in range(4)}
                for i, val in enumerate(four_values):
                    data_dict[condition][i].append(val)

    # Process and print results, and plot
    for condition, values in data_dict.items():
        print(f"Results for condition: {condition}")
        fig, axes = plt.subplots(2, 2)  # Create subplots for the four lines
        for i in range(4):
            ax = axes[i // 2, i % 2]  # Select the subplot
            averages = values[i]

            # Apply smoothing if applicable
            if smoothing > 1:
                averages = smooth(averages, smoothing)

            # Plot the results for this element
            ax.plot(averages, label=f"Element {i}")
            ax.set_title(f"Element {i}")
            for change_point in change_points:
                ax.axvline(x=change_point, color="grey", linestyle="--")

        # Add labels to the subplots
        fig.text(0.5, 0.04, "Index", ha="center")
        fig.text(0.04, 0.5, "Value", va="center", rotation="vertical")
        plt.suptitle(f"Results for condition: {condition}")
        plt.show()


# Example usage
process_files_with_list(file_pattern="replace*.txt", smoothing=1)
