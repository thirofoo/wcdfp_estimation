import pandas as pd
import matplotlib.pyplot as plt

output_prefix = "src/evaluation/output"
prefix_template = "src/evaluation/output/{task_count}_{wcdfp}_{epsilon}"

def plot_execution_time_boxplot():
    """
    Generate a grid of boxplots showing execution time distribution
    for different task counts (10 to 100) and utilization thresholds (0.6, 0.65, 0.7).
    The x-axis represents task counts, and the y-axis represents utilization thresholds.
    Each boxplot shows the execution time distribution for all methods, with method names grouped.
    """
    task_counts = range(10, 101, 10)  # Task counts from 10 to 100 in steps of 10
    util_rate_values = [0.6, 0.65, 0.7]  # Utilization thresholds
    epsilon = "0.001"  # Fixed epsilon value

    method_names = ["berry_essen", "convolution_doubling", "convolution", "monte_carlo"]

    fig, axes = plt.subplots(len(util_rate_values), len(task_counts), figsize=(30, 15), sharex=True, sharey=True)

    for row_idx, wcdfp in enumerate(util_rate_values):
        for col_idx, task_count in enumerate(task_counts):
            ax = axes[row_idx, col_idx]

            # Combine execution times for all methods
            combined_execution_times = []
            for method_name in method_names:
                prefix = prefix_template.format(task_count=task_count, wcdfp=wcdfp, epsilon=epsilon)
                csv_path = f"{prefix}/evaluation_{method_name.lower()}.csv"
                try:
                    data = pd.read_csv(csv_path)
                    if "ExecutionTime" not in data.columns:
                        raise ValueError(f"The file {csv_path} does not contain an 'ExecutionTime' column.")

                    filtered_times = data["ExecutionTime"][data["ExecutionTime"] >= 0]
                    combined_execution_times.append(filtered_times)
                except FileNotFoundError:
                    print(f"File not found: {csv_path}")
                    combined_execution_times.append([])

            # Plot boxplot for the combined execution times
            ax.boxplot(
                combined_execution_times,
                labels=None,  # Remove individual method labels to declutter
                vert=True,
                patch_artist=True,
                whis=[0, 100]
            )
            ax.set_yscale("log")
            ax.grid(axis="y", linestyle="--", linewidth=0.5)

            if row_idx == len(util_rate_values) - 1:
                ax.set_xlabel(f"{task_count}\n({', '.join([str(i + 1) for i in range(len(method_names))])})", fontsize=12)  # Task count with method indices
            if col_idx == 0:
                ax.set_ylabel(f"Utilization: {wcdfp}", fontsize=12)

    # Add a single x-axis and y-axis label
    fig.text(0.5, 0.02, "Task Count (Methods: 1. berry_essen, 2. convolution_doubling, 3. convolution, 4. monte_carlo)", ha="center", va="center", fontsize=16)
    fig.text(0.02, 0.5, "Utilization", ha="center", va="center", rotation="vertical", fontsize=16)

    # Adjust layout and save the figure
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    output_path = f"{output_prefix}/execution_time_boxplot_high_res.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"High-resolution boxplot saved to {output_path}")


def plot_wcdfp_comparison():
    """
    Plot all WCDFP data points for each method pair across all task counts and utilization thresholds
    and generate comparison plots with y = x as a reference line.
    """
    task_counts = range(10, 11, 10)  # Task counts from 10 to 100 in steps of 10
    util_rate_values = [0.6]  # Utilization thresholds
    epsilon = "0.001"  # Fixed epsilon value
    method_names = ["berry_essen", "convolution_doubling", "convolution", "monte_carlo"]

    for i in range(len(method_names)):
        for j in range(i + 1, len(method_names)):
            method_1 = method_names[i]
            method_2 = method_names[j]

            wcdfp_data = []
            for task_count in task_counts:
                for wcdfp in util_rate_values:
                    util_rate_values_per_method = {}
                    for method_name in [method_1, method_2]:
                        prefix = prefix_template.format(task_count=task_count, wcdfp=wcdfp, epsilon=epsilon)
                        csv_path = f"{prefix}/evaluation_{method_name.lower()}.csv"
                        try:
                            data = pd.read_csv(csv_path)
                            if "WCDFP" not in data.columns:
                                raise ValueError(f"The file {csv_path} does not contain a 'WCDFP' column.")
                            util_rate_values_per_method[method_name] = data["WCDFP"].values
                        except FileNotFoundError:
                            print(f"File not found: {csv_path}")
                            util_rate_values_per_method[method_name] = None

                    wcdfp_1 = util_rate_values_per_method.get(method_1)
                    wcdfp_2 = util_rate_values_per_method.get(method_2)

                    if method_2 == "monte_carlo" and method_1 == "convolution" and len(wcdfp_1[wcdfp_2 < wcdfp_1]) > 0:
                        idx = wcdfp_2 < wcdfp_1
                        print(f"Task Count: {task_count}, Utilization: {wcdfp}, Method 1: {method_1}, Method 2: {method_2}")
                        print(f"WCDFP (Convolution): {wcdfp_1[idx]}")
                        print(f"WCDFP (Monte Carlo): {wcdfp_2[idx]}")

                    if wcdfp_1 is not None and wcdfp_2 is not None:
                        min_len = min(len(wcdfp_1), len(wcdfp_2))
                        if len(wcdfp_1) != len(wcdfp_2):
                            print("Warning: Utilization values are not equal")
                            print(f"len(wcdfp_1): {len(wcdfp_1)}, len(wcdfp_2): {len(wcdfp_2)}")
                        wcdfp_data.extend(zip(wcdfp_1[:min_len], wcdfp_2[:min_len]))

            # Extract values for plotting
            x_vals = [x[0] for x in wcdfp_data]
            y_vals = [x[1] for x in wcdfp_data]

            # Create the plot
            plt.figure(figsize=(8, 8))
            plt.scatter(x_vals, y_vals, color="orange", label="WCDFP Points")
            min_val = min(min(x_vals), min(y_vals))
            max_val = max(max(x_vals), max(y_vals))
            plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="blue", label="y = x (Reference Line)")

            # Configure plot
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel(f"{method_1} WCDFP (log scale)")
            plt.ylabel(f"{method_2} WCDFP (log scale)")
            plt.title(f"WCDFP Ratios: {method_1} vs {method_2}")
            plt.legend()
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)

            # Save the plot
            output_path = f"{output_prefix}/comparison_ratios_{method_1.lower()}_{method_2.lower()}.png"
            plt.savefig(output_path, dpi=300)
            plt.close()
            print(f"Plot saved to {output_path}")
