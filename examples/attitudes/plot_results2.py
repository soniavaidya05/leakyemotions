import glob
import numpy as np
import matplotlib.pyplot as plt


def process_files(file_pattern="*.txt", smoothing=1, time_range=None):
    # Dictionary to store data
    data_dict = {}
    change_points = set()

    # Function for smoothing
    def smooth(data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode="valid")

    # Function to plot results
    def plot_results(time_steps, averages, stderrs, label):
        plt.errorbar(time_steps[: len(averages)], averages, yerr=stderrs, label=label)

    # Iterate over each file matching the pattern
    prev_condition = None
    for filename in glob.glob(file_pattern):
        with open(filename, "r") as file:
            for line in file:
                # Check if line starts with a number and has at least 4 parts
                parts = line.split()
                if len(parts) >= 4 and parts[0].isdigit():
                    time_step = int(parts[0])
                    third_integer = int(parts[2])
                    last_string = parts[-1]
                    condition = parts[-2]  # second to last column

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
                    if last_string not in data_dict:
                        data_dict[last_string] = {}
                    if time_step not in data_dict[last_string]:
                        data_dict[last_string][time_step] = []
                    data_dict[last_string][time_step].append(third_integer)

    # Process and print results, and plot
    for key, value in data_dict.items():
        print(f"Results for {key}:")
        time_steps = sorted(value.keys())
        averages = []
        stderrs = []
        for t in time_steps:
            mean = np.mean(value[t])
            stderr = np.std(value[t]) / np.sqrt(len(value[t]))
            averages.append(mean)
            stderrs.append(stderr)

        # Apply smoothing if applicable
        if smoothing > 1:
            averages = smooth(averages, smoothing)
            stderrs = smooth(stderrs, smoothing)

        for t, avg, err in zip(time_steps[: len(averages)], averages, stderrs):
            print(f"Time step: {t}, Average: {avg}, Standard Error: {err}")

        plot_results(time_steps, averages, stderrs, key)
        print()

    # Draw vertical lines at change points
    for change_point in change_points:
        plt.axvline(x=change_point, color="grey", linestyle="--")

    # Add legend and labels, and show plot
    plt.legend()
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.title("Results")
    plt.show()


# Example usage for time steps between 100 and 300
# process_files(smoothing=5, time_range=(1500, 3000))
# process_files(file_pattern="study1_*.txt", smoothing=5)
# process_files(file_pattern="study2b_*.txt", smoothing=5)
# process_files(file_pattern="test*.txt", smoothing=5)
process_files(file_pattern="weighted2_*.txt", smoothing=5)
