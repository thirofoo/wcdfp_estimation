import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import argparse
import sys
import os

output_prefix = "src/evaluation/output"
prefix_template = "src/evaluation/output/{task_count}_{util_rate:.2f}_{epsilon}"


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

    # method_names = ["berry_essen", "convolution_doubling", "convolution", "monte_carlo"]
    method_names = ["berry_essen", "convolution_doubling", "convolution", "monte_carlo", "convolution_doubling_float128", "convolution_float128"]

    fig, axes = plt.subplots(len(util_rate_values), len(task_counts), figsize=(30, 15), sharex=True, sharey=True)

    for row_idx, util_rate in enumerate(util_rate_values):
        for col_idx, task_count in enumerate(task_counts):
            ax = axes[row_idx, col_idx]

            # Combine execution times for all methods
            combined_execution_times = []
            for method_name in method_names:
                prefix = prefix_template.format(task_count=task_count, util_rate=util_rate, epsilon=epsilon)
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
                ax.set_ylabel(f"Utilization: {util_rate}", fontsize=12)

    # Add a single x-axis and y-axis label
    fig.text(0.5, 0.02, "Task Count (Methods: 1. berry_essen, 2. convolution_doubling, 3. convolution, 4. monte_carlo)", ha="center", va="center", fontsize=16)
    fig.text(0.02, 0.5, "Utilization", ha="center", va="center", rotation="vertical", fontsize=16)

    # Adjust layout and save the figure
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    output_path = f"{output_prefix}/execution_time_boxplot_high_res.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"High-resolution boxplot saved to {output_path}")


def plot_time_ratio_vs_wcdfp_ratio(mode=0):
    """
    Plot ExecutionTime ratios vs WCDFP ratios for each method pair across all task counts and utilization thresholds.
    Generate comparison plots with reference lines x=1 and y=1, centered around (1, 1).

    :param mode: Determines the coloring mode for the plot.
                 0 - No color gradient.
                 1 - Gradient based on utilization rate.
                 2 - Gradient based on task count.
    """
    task_counts = range(10, 101, 10)  # Task counts from 10 to 40 in steps of 10
    util_rate_values = [0.6, 0.65, 0.70]  # Utilization thresholds
    epsilon = "0.001"  # Fixed epsilon value
    method_names = ["berry_essen", "convolution_doubling", "convolution", "monte_carlo"]
    # method_names = ["convolution_doubling", "convolution", "convolution_doubling_float128", "convolution_float128"]

    # Parse command-line arguments
    args = sys.argv[1:]
    if '--' in args:
        args.remove('--')

    parser = argparse.ArgumentParser(description="Plot WCDFP comparison.")
    parser.add_argument("--mode", type=int, default=0, help="Coloring mode: 0 (no gradient), 1 (utilization), 2 (task count)")
    parsed_args = parser.parse_args(args)
    mode = parsed_args.mode
    print(f"Selected mode: {mode}")

    # Define custom colormaps
    custom_YlGn = LinearSegmentedColormap.from_list("custom_YlGn", ["#9ACD32", "#32CD32", "#006400"])
    custom_Blues = LinearSegmentedColormap.from_list("custom_Blues", ["#1E90FF", "#4682B4", "#00008B"])

    for i in range(len(method_names)):
        for j in range(i + 1, len(method_names)):
            method_1 = method_names[i]
            method_2 = method_names[j]

            ratio_data = []
            color_values = []

            # Define gradient calculation ranges
            if mode == 1:
                color_min, color_max = min(util_rate_values), max(util_rate_values)
                color_label = "Utilization Rate"
            elif mode == 2:
                color_min, color_max = min(task_counts), max(task_counts)
                color_label = "Task Count"
            else:
                color_min, color_max = None, None

            for task_count in task_counts:
                for util_rate in util_rate_values:
                    metrics_per_method = {}

                    for method_name in [method_1, method_2]:
                        prefix = prefix_template.format(task_count=task_count, util_rate=util_rate, epsilon=epsilon)
                        csv_path = os.path.join(prefix, f"evaluation_{method_name.lower()}.csv")
                        try:
                            data = pd.read_csv(csv_path)
                            if "WCDFP" not in data.columns or "ExecutionTime" not in data.columns:
                                raise ValueError(f"File {csv_path} missing required columns.")
                            metrics_per_method[method_name] = {
                                "WCDFP": data["WCDFP"].values,
                                "ExecutionTime": data["ExecutionTime"].values
                            }
                        except FileNotFoundError:
                            print(f"File not found: {csv_path}")
                            metrics_per_method[method_name] = None

                    if metrics_per_method[method_1] and metrics_per_method[method_2]:
                        wcdfp_1 = metrics_per_method[method_1]["WCDFP"]
                        wcdfp_2 = metrics_per_method[method_2]["WCDFP"]
                        time_1 = metrics_per_method[method_1]["ExecutionTime"]
                        time_2 = metrics_per_method[method_2]["ExecutionTime"]

                        min_len = min(len(wcdfp_1), len(wcdfp_2))
                        for k in range(min_len):
                            if time_1[k] < 0 or time_2[k] < 0:
                                print(f"Negative execution time detected for {method_1} or {method_2}. Skipping plot.")
                                print(f"Task Count: {task_count}, Utilization Rate: {util_rate}")
                                continue

                            time_ratio = time_1[k] / time_2[k]
                            wcdfp_ratio = wcdfp_1[k] / wcdfp_2[k]
                            ratio_data.append((time_ratio, wcdfp_ratio))

                            # Assign color value based on mode
                            if mode in [1, 2]:
                                color_key = util_rate if mode == 1 else task_count
                                normalized_color = (color_key - color_min) / (color_max - color_min)
                                color_values.append(normalized_color)

            if not ratio_data:
                print(f"No data available for {method_1} vs {method_2}. Skipping plot.")
                continue

            x_vals = [x[0] for x in ratio_data]
            y_vals = [x[1] for x in ratio_data]

            # Create the scatter plot
            fig, ax = plt.subplots(figsize=(8, 8))
            if mode in [1, 2]:
                cmap = custom_YlGn if mode == 1 else custom_Blues
                scatter = ax.scatter(x_vals, y_vals, c=color_values, cmap=cmap, vmin=0, vmax=1, label="Ratios")
            else:
                scatter = ax.scatter(x_vals, y_vals, c="orange", label="Ratios")

            # Add reference lines x=1 and y=1
            ax.axvline(x=1, color="gray", linestyle="--", linewidth=1, label="x = 1 (Equal Execution Time)")
            ax.axhline(y=1, color="gray", linestyle="--", linewidth=1, label="y = 1 (Equal WCDFP)")

            # Log scales
            ax.set_xscale("log")
            ax.set_yscale("log")

            # Adjust limits so (1, 1) is at the center
            margin_lower = 0.1
            margin_upper = 1.0 / margin_lower
            min_x, min_y = min(x_vals) * margin_lower, min(y_vals) * margin_lower
            max_x, max_y = max(x_vals) * margin_upper, max(y_vals) * margin_upper
            using_x_abs = max(1 / min_x, max_x)
            using_y_abs = max(1 / min_y, max_y)
            print(f"X: {min_x} - {max_x}, Y: {min_y} - {max_y}")
            ax.set_xlim([1 / using_x_abs, using_x_abs])
            ax.set_ylim([1 / using_y_abs, using_y_abs])

            # Labels and title
            ax.set_xlabel(f"Execution Time Ratio ({method_1} / {method_2})", fontsize=12)
            ax.set_ylabel(f"WCDFP Ratio ({method_1} / {method_2})", fontsize=12)
            ax.set_title(f"Time vs WCDFP Ratio: {method_1} vs {method_2}", fontsize=14)
            ax.legend()
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
            ax.set_box_aspect(1)

            # Color bar
            if mode in [1, 2]:
                cbar = fig.colorbar(scatter, ax=ax, aspect=40, shrink=0.8, pad=0.02)
                # cbar.set_label(color_label, rotation=270, labelpad=15)

                cbar.set_ticks([0, 1])
                cbar.set_ticklabels([f"{color_min}", f"{color_max}"])

            # Save the plot
            output_path = os.path.join(output_prefix, f"time-wcdfp-ratio-{method_1.lower()}-{method_2.lower()}.png")
            plt.tight_layout(rect=[0.02, 0.02, 0.95, 0.95])
            plt.savefig(output_path, dpi=300)
            plt.close()
            print(f"Plot saved to {output_path}")


def plot_wcdfp_comparison(mode=0):
    """
    Plot all WCDFP data points for each method pair across all task counts and utilization thresholds
    and generate comparison plots with y = x as a reference line.

    :param mode: Determines the coloring mode for the plot (default=0).
                 0 - No color gradient.
                 1 - Gradient based on utilization rate.
                 2 - Gradient based on task count.
    """
    task_counts = range(10, 101, 10)  # Task counts from 10 to 100 in steps of 10
    util_rate_values = [0.6, 0.65, 0.70]  # Utilization thresholds
    epsilon = "0.001"  # Fixed epsilon value
    method_names = ["berry_essen", "convolution_doubling", "convolution", "monte_carlo"]
    # method_names = ["convolution_doubling", "convolution", "convolution_doubling_float128", "convolution_float128"]

    # Parse command-line arguments
    args = sys.argv[1:]
    if '--' in args:
        args.remove('--')

    parser = argparse.ArgumentParser(description="Plot WCDFP comparison.")
    parser.add_argument("--mode", type=int, default=0, help="Coloring mode: 0 (no gradient), 1 (utilization), 2 (task count)")
    parsed_args = parser.parse_args(args)
    mode = parsed_args.mode
    print(f"Selected mode: {mode}")

    # Define custom colormaps
    custom_YlGn = LinearSegmentedColormap.from_list(
        "custom_YlGn",
        ["#9ACD32", "#32CD32", "#006400"]  # 黄緑から濃い緑へ
    )

    custom_Blues = LinearSegmentedColormap.from_list(
        "custom_Blues",
        ["#1E90FF", "#4682B4", "#00008B"]  # 水色から濃い青へ
    )

    for i in range(len(method_names)):
        for j in range(i + 1, len(method_names)):
            method_1 = method_names[i]
            method_2 = method_names[j]

            wcdfp_data = []
            color_values = []

            # Define minimum and maximum for gradient calculation
            if mode == 1:
                color_min, color_max = min(util_rate_values), max(util_rate_values)
                color_label = "Utilization Rate"
            elif mode == 2:
                color_min, color_max = min(task_counts), max(task_counts)
                color_label = "Task Count"
            else:
                color_min, color_max = None, None

            for task_count in task_counts:
                for util_rate in util_rate_values:
                    util_rate_values_per_method = {}
                    for method_name in [method_1, method_2]:
                        prefix = prefix_template.format(task_count=task_count, util_rate=util_rate, epsilon=epsilon)
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

                    if wcdfp_1 is not None and wcdfp_2 is not None:
                        min_len = min(len(wcdfp_1), len(wcdfp_2))
                        wcdfp_data.extend(zip(wcdfp_1[:min_len], wcdfp_2[:min_len]))

                        if mode in [1, 2]:
                            # Calculate normalized color value (0 to 1)
                            color_key = util_rate if mode == 1 else task_count
                            normalized_color = (color_key - color_min) / (color_max - color_min)
                            # クランプ処理を削除し、normalized_color をそのまま使用
                            color_values.extend([normalized_color] * min_len)

            if not wcdfp_data:
                print(f"No data available for {method_1} vs {method_2}. Skipping plot.")
                continue

            x_vals = [x[0] for x in wcdfp_data]
            y_vals = [x[1] for x in wcdfp_data]

            # Create plot with square aspect ratio and color bar outside
            fig, ax = plt.subplots(figsize=(8, 8))
            if mode in [1, 2]:
                if mode == 1:
                    # 黄緑から濃い緑へのカラーマップを使用
                    cmap = custom_YlGn
                elif mode == 2:
                    # 水色から濃い青へのカラーマップを使用
                    cmap = custom_Blues

                scatter = ax.scatter(
                    x_vals,
                    y_vals,
                    c=color_values,
                    cmap=cmap,  # カスタムカラーマップを使用
                    label="WCDFP Points",
                    vmin=0,
                    vmax=1
                )
            else:
                scatter = ax.scatter(
                    x_vals,
                    y_vals,
                    c="orange",
                    label="WCDFP Points"
                )

            min_val = min(min(x_vals), min(y_vals))
            max_val = max(max(x_vals), max(y_vals))
            ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="blue", label="y = x (Reference Line)")

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(f"{method_1} WCDFP (log scale)")
            ax.set_ylabel(f"{method_2} WCDFP (log scale)")
            ax.set_title(f"WCDFP Ratios: {method_1} vs {method_2}")
            ax.legend()
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)

            # Force equal aspect ratio
            ax.set_aspect('equal', adjustable='box')

            # Add color bar outside the plot
            if mode in [1, 2]:
                cbar = fig.colorbar(scatter, ax=ax, aspect=40, shrink=0.8, pad=0.02)
                cbar.set_label(color_label, rotation=270, labelpad=15)
                # Set ticks and labels based on color_min and color_max
                cbar.set_ticks([0, 1])
                cbar.set_ticklabels([f"{color_min}", f"{color_max}"])

            # Save the plot
            plt.tight_layout(rect=[0.02, 0.02, 0.95, 0.95])  # Adjust layout for square plot
            output_path = f"{output_prefix}/comparison-ratios-{method_1.lower()}-{method_2.lower()}.png"
            plt.savefig(output_path, dpi=300)
            plt.close()
            print(f"Plot saved to {output_path}")


def plot_comparison_for_task_id():
    """
    Plot ExecutionTime vs WCDFP for a given TaskSetID, comparing 'adjust_sample', 'berry_essen', and 'convolution'.

    :param task_id: TaskSetID to filter the data and plot results.
    """
    task_id = 1
    task_num = 50
    utilization_rate = 0.60
    epsilon = "0.001"

    # Define the directory path
    directory = f"{output_prefix}/{task_num}_{utilization_rate:.2f}_{epsilon}"
    methods = ["monte_carlo_adjust_sample", "monte_carlo_adjust_sample_single", "berry_essen", "convolution", "convolution_doubling"]

    # Define colors for each method
    method_colors = {
        "monte_carlo_adjust_sample": "tab:blue",  # Line + Points
        "monte_carlo_adjust_sample_single": "tab:purple",  # Line + Points
        "berry_essen": "tab:orange",                      # Points
        "convolution": "tab:green",                       # Points
        "convolution_doubling": "tab:red"
    }

    # Initialize plot
    plt.figure(figsize=(10, 7))

    for method in methods:
        if method == "monte_carlo_adjust_sample" or method == "monte_carlo_adjust_sample_single":
            input_file = f"evaluation_{method}_{task_id}.csv"
        else:
            input_file = f"evaluation_{method}.csv"
        file_path = os.path.join(directory, input_file)
        
        # Check if the file exists
        if not os.path.isfile(file_path):
            print(f"File not found: {file_path}")
            continue

        # Load data
        df = pd.read_csv(file_path)

        # Filter data by TaskSetID
        df = df[df["TaskSetID"] == task_id]

        # Extract relevant columns
        x = df["ExecutionTime"]
        y = df["WCDFP"]

        if method == "monte_carlo_adjust_sample" or method == "monte_carlo_adjust_sample_single":
            # For adjust_sample, plot a line with points
            plt.plot(x, y, marker='o', linestyle='-', color=method_colors[method],
                     label=f"{method}")
        elif method == "convolution":
            # For convolution, plot points
            plt.scatter(x, y, color=method_colors[method], s=60, label="Convolution")

            # Add a dotted horizontal line for each WCDFP value
            for i, wcdfp_value in enumerate(y):
                plt.axhline(y=wcdfp_value, xmin=0, xmax=1, color=method_colors[method],
                            linestyle='--', linewidth=1, alpha=0.7)
        else:
            # For other methods, plot points only
            plt.scatter(x, y, color=method_colors[method], s=60, label=method.replace("_", " ").title())

    # Set log scale for both axes
    plt.xscale("log")
    plt.yscale("log")

    # Add labels and title
    plt.xlabel("Execution Time (log scale) [s]", fontsize=12)
    plt.ylabel("WCDFP (log scale)", fontsize=12)
    plt.title(f"Comparison of WCDFP vs Execution Time (TaskSetID={task_id})", fontsize=14)

    # Add grid and legend
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend()

    # Save plot directly to output_prefix
    output_path = os.path.join(output_prefix, f"comparison_taskset_{task_id}.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to {output_path}")
