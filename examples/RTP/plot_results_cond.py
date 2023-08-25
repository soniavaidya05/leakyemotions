import glob
import numpy as np
import matplotlib.pyplot as plt
import argparse
import re


# Function to plot results
def process_files(file_pattern="*.txt", time_range=None):
    # Dictionary to store data
    data_dict = {}

    # Compile regex pattern to extract needed data
    pattern = re.compile(
        r"(\d+)\s+\d+\s+\d+\s+(\[\d+,\s*\d+,\s*\d+\])\s+\d+(\.\d+)?\s+(\[\d+,\s*\d+,\s*\d+,\s*\d+\])\s+\d+(\.\d+)?\s+\d+\s+(\S+)$"
    )

    # Iterate over each file matching the pattern


for filename in glob.glob(file_pattern):
    with open(filename, "r") as file:
        for line in file:
            parts = line.split()
            if len(parts) >= 4 and parts[0].isdigit():
                time_step = int(parts[0])

                # Error-handling for second_list
                try:
                    second_list = eval(
                        parts[3]
                    )  # Evaluating the string to get the list
                    if not isinstance(second_list, list):
                        print(
                            f"Skipped malformed line in file {filename}: {line.strip()}"
                        )
                        continue
                except (SyntaxError, IndexError):
                    print(f"Skipped malformed line in file {filename}: {line.strip()}")
                    continue

                condition = parts[-1]

    # Plot results
    for key, value in data_dict.items():
        time_steps = sorted(value.keys())
        values = [value[t] for t in time_steps]
        plot_results(time_steps, values, key, ylabel="List Elements")


def plot_results(time_steps, values, label, ylabel="Value"):
    plt.figure()
    for idx, val_series in enumerate(np.array(values).T):
        plt.plot(time_steps[: len(val_series)], val_series, label=f"Element {idx+1}")
    plt.legend()
    plt.xlabel("Time Step")
    plt.ylabel(ylabel)
    plt.title(f"Results for {label}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot data from text files.")
    parser.add_argument(
        "-p",
        "--pattern",
        type=str,
        default="*.txt",
        help="File pattern to match. Default is '*.txt'.",
    )
    parser.add_argument(
        "-r",
        "--range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Time range to consider, given as two integers: START END. If not provided, all times are considered.",
    )
    args = parser.parse_args()

    # Extract values from the parsed arguments
    file_pattern = args.pattern
    time_range = tuple(args.range) if args.range else None

    # Process files and plot data
    process_files(file_pattern=file_pattern, time_range=time_range)


if __name__ == "__main__":
    main()
