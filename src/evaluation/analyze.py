import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_wcdfp_comparison():
    # method_name_1 = "berry_essen"
    # method_name_1 = "monte_carlo"
    method_name_1 = "convolution"
    method_name_2 = "convolution_doubling"
    prefix = "src/evaluation/output/100_0.65_0.001"
    csv_path_1 = f"{prefix}/evaluation_{method_name_1.lower()}.csv"
    csv_path_2 = f"{prefix}/evaluation_{method_name_2.lower()}.csv"
    output_path = f"{prefix}/wcdfp_comparison_{method_name_1.lower()}_{method_name_2.lower()}.png"

    # Read CSV files
    data_1 = pd.read_csv(csv_path_1)
    data_2 = pd.read_csv(csv_path_2)

    # Merge data on TaskSetID
    merged_data = pd.merge(
        data_1[["TaskSetID", "WCDFP"]],
        data_2[["TaskSetID", "WCDFP"]],
        on="TaskSetID",
        suffixes=(f"_{method_name_1.lower()}", f"_{method_name_2.lower()}")
    )

    # Extract WCDFP values
    wcdfp_1 = merged_data[f"WCDFP_{method_name_1.lower()}"]
    wcdfp_2 = merged_data[f"WCDFP_{method_name_2.lower()}"]

    # Create the plot
    plt.figure(figsize=(8, 8))
    plt.scatter(wcdfp_1, wcdfp_2, color="orange", label="WCDFP Points")
    plt.plot([min(wcdfp_1.min(), wcdfp_2.min()), max(wcdfp_1.max(), wcdfp_2.max())], 
             [min(wcdfp_1.min(), wcdfp_2.min()), max(wcdfp_1.max(), wcdfp_2.max())],
             color="blue", linestyle="--", label="y = x (Reference Line)")  # Reference line y=x

    # Configure plot
    plt.xscale("log")  # Set x-axis to log scale
    plt.yscale("log")  # Set y-axis to log scale
    plt.xlabel(f"{method_name_1} WCDFP (log scale)")
    plt.ylabel(f"{method_name_2} WCDFP (log scale)")
    plt.title(f"WCDFP Comparison: {method_name_1} vs {method_name_2}")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.axis("equal")  # Ensure equal scaling for x and y axes

    # Save the plot
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")


def plot_execution_time_boxplot():
    method_names = ["berry_essen", "convolution_doubling", "monte_carlo"]
    prefix = "src/evaluation/output/100_0.65_0.001"
    
    # Generate CSV paths for each method name
    csv_paths = [f"{prefix}/evaluation_{method_name.lower()}.csv" for method_name in method_names]
    output_path = f"{prefix}/execution_time_distribution.png"

    # Read execution times from each CSV file
    execution_times = []
    for csv_path in csv_paths:
        data = pd.read_csv(csv_path)
        if "ExecutionTime" not in data.columns:
            raise ValueError(f"The file {csv_path} does not contain an 'ExecutionTime' column.")
        
        filtered_times = data["ExecutionTime"][data["ExecutionTime"] >= 0]
        execution_times.append(filtered_times)

    # Create the boxplot
    plt.figure(figsize=(10, 6))
    # plt.boxplot(execution_times, labels=method_names, vert=True, patch_artist=True)
    plt.boxplot(execution_times, labels=method_names, vert=True, patch_artist=True, whis=[0, 100])
    plt.yscale("log")  # Set log scale for Y-axis

    # Configure plot
    plt.xlabel("Methods")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Execution Time Distribution Across Methods")
    plt.grid(axis="y", linestyle="--", linewidth=0.5)

    # Save the plot
    plt.savefig(output_path)
    plt.close()
    print(f"Boxplot saved to {output_path}")
