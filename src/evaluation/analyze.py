import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import fitz

# Global configuration
OUTPUT_PREFIX = "src/evaluation/output"
PREFIX_TEMPLATE = os.path.join("src", "evaluation", "output", "{task_count}_{util_rate:.2f}_{epsilon}")

METHOD_NAMES = [
    "berry_essen",
    "aggregate_conv_imp",
    "sequential_conv",
    "monte_carlo",
    # "aggregate_conv_orig",
    # "aggregate_conv_orig_float128",
    # "sequential_conv_float128"
]

METHOD_TO_LABEL = {
    "monte_carlo": "MC",
    "berry_essen": "BE",
    "monte_carlo_multi": "MC (24 thread)",
    "monte_carlo_single": "MC (1 thread)",
    "sequential_conv": "SC",
    "aggregate_conv_imp": "AC",
    # "aggregate_conv_orig": "AC \ (Orig.)",
    # "aggregate_conv_imp": "AC \ (Imp.)",
    # "aggregate_conv_orig_float128": "RC (float128)",
    # "sequential_conv_float128": "SC (float128)"
}

# Constant parameter values for most plots
TASK_COUNTS = range(10, 101, 10)
UTIL_RATE_VALUES = [0.6, 0.65, 0.70]
EPSILON = "0.001"


def get_prefix(task_count, util_rate):
    return PREFIX_TEMPLATE.format(task_count=task_count, util_rate=util_rate, epsilon=EPSILON)


def load_csv_data(file_path, columns=None):
    """
    Returns a DataFrame for a given file path. Optionally checks for required columns.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    if columns:
        for col in columns:
            if col not in df.columns:
                print(f"The file {file_path} does not contain '{col}' column.")
                return None
    return df


def plot_execution_time_boxplot():
    """
    Aggregates execution times across parameter values and produces a horizontal log-scale boxplot.
    """
    all_methods_times = []
    for method in METHOD_NAMES:
        method_times = []
        for task_count in TASK_COUNTS:
            for util_rate in UTIL_RATE_VALUES:
                prefix = get_prefix(task_count, util_rate)
                csv_path = os.path.join(prefix, f"evaluation_{method.lower()}.csv")
                df = load_csv_data(csv_path, columns=["ExecutionTime"])
                if df is None:
                    continue
                # Only non-negative data
                valid_times = df["ExecutionTime"][df["ExecutionTime"] >= 0].tolist()
                method_times.extend(valid_times)
        all_methods_times.append(method_times)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(all_methods_times, patch_artist=True, whis=[0, 100],
               showfliers=True, vert=False, widths=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Execution Time [s]", fontsize=20, labelpad=16)
    ax.set_ylim(0.5, len(METHOD_NAMES) + 0.5)
    ax.margins(y=0.0)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    ax.set_yticks(range(1, len(METHOD_NAMES) + 1))
    ax.set_yticklabels([f"$\\mathbf{{{METHOD_TO_LABEL[m]}}}$" for m in METHOD_NAMES],
                       va="center", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(axis="x", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_PREFIX, "execution_time_boxplot_aggregated.pdf")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Aggregated horizontal boxplot saved to {output_path}")


def get_color_params(mode):
    if mode == 1:
        return min(UTIL_RATE_VALUES), max(UTIL_RATE_VALUES), "Utilization Rate"
    elif mode == 2:
        return min(TASK_COUNTS), max(TASK_COUNTS), "Task Count"
    return None, None, None


def plot_scatter_for_methods(method_1, method_2, ax, mode, custom_cmap):
    """
    Helper function to create the scatter plot for two methods.
    """
    ratio_data = []
    color_values = []

    color_min, color_max, _ = get_color_params(mode)

    for task_count in TASK_COUNTS:
        for util_rate in UTIL_RATE_VALUES:
            metrics = {}
            for method in [method_1, method_2]:
                prefix = get_prefix(task_count, util_rate)
                csv_path = os.path.join(prefix, f"evaluation_{method.lower()}.csv")
                df = load_csv_data(csv_path, columns=["WCDFP", "ExecutionTime"])
                metrics[method] = df

            if metrics[method_1] is None or metrics[method_2] is None:
                continue

            wcdfp_1, time_1 = metrics[method_1]["WCDFP"].values, metrics[method_1]["ExecutionTime"].values
            wcdfp_2, time_2 = metrics[method_2]["WCDFP"].values, metrics[method_2]["ExecutionTime"].values
            min_len = min(len(wcdfp_1), len(wcdfp_2))
            for k in range(min_len):
                ratio_data.append((time_1[k] / time_2[k], wcdfp_1[k] / wcdfp_2[k]))
                if mode in [1, 2]:
                    color_key = util_rate if mode == 1 else task_count
                    norm_color = (color_key - color_min) / (color_max - color_min)
                    color_values.append(norm_color)

    if not ratio_data:
        return None

    x_vals = [v[0] for v in ratio_data]
    y_vals = [v[1] for v in ratio_data]
    scatter = ax.scatter(x_vals, y_vals,
                         c=color_values if mode in [1, 2] else "orange",
                         cmap=custom_cmap, vmin=0, vmax=1, s=5)
    # Draw reference lines
    ax.axvline(x=1, linestyle="--", color="#0F4D48", zorder=10)
    ax.axhline(y=1, linestyle="--", color="#0F4D48", zorder=10)

    # Count points in quadrants
    quad_counts = {
        "Q1": sum(1 for x, y in zip(x_vals, y_vals) if x > 1 and y > 1),
        "Q2": sum(1 for x, y in zip(x_vals, y_vals) if x <= 1 and y > 1),
        "Q3": sum(1 for x, y in zip(x_vals, y_vals) if x <= 1 and y <= 1),
        "Q4": sum(1 for x, y in zip(x_vals, y_vals) if x > 1 and y <= 1),
    }
    ax.text(0.02, 0.98, f"{quad_counts['Q2']}",
            transform=ax.transAxes, fontsize=12, ha='left', va='top',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle="round,pad=0.3"))
    ax.text(0.98, 0.98, f"{quad_counts['Q1']}",
            transform=ax.transAxes, fontsize=12, ha='right', va='top',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle="round,pad=0.3"))
    ax.text(0.02, 0.02, f"{quad_counts['Q3']}",
            transform=ax.transAxes, fontsize=12, ha='left', va='bottom',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle="round,pad=0.3"))
    ax.text(0.98, 0.02, f"{quad_counts['Q4']}",
            transform=ax.transAxes, fontsize=12, ha='right', va='bottom',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle="round,pad=0.3"))

    # Set axes limits and scale
    margin = 0.1
    using_x = max(1 / (min(x_vals) * margin), max(x_vals) * (1 / margin))
    using_y = max(1 / (min(y_vals) * margin), max(y_vals) * (1 / margin))
    ax.set_xlim([1 / using_x / 10, using_x * 10])
    ax.set_ylim([1 / using_y / 10, using_y * 10])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"Execution Time Ratio ($\\bf{{{METHOD_TO_LABEL[method_1]}}} / \\bf{{{METHOD_TO_LABEL[method_2]}}}$)", fontsize=16)
    ax.set_ylabel(f"WCDFP Ratio ($\\bf{{{METHOD_TO_LABEL[method_1]}}} / \\bf{{{METHOD_TO_LABEL[method_2]}}}$)", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(visible=True, which='major', linestyle='--', linewidth=0.5)
    ax.set_box_aspect(1)
    return scatter


def plot_time_ratio_vs_wcdfp_ratio(mode=0, rows=2, cols=3, output_file="merged_ratio_plot.pdf"):
    """
    Merge Time Ratio vs WCDFP Ratio scatter plots into a grid layout.
    Mode 0: no color gradient, 1: gradient by utilization, 2: gradient by task count.
    """
    parser = argparse.ArgumentParser(description="Plot Time Ratio vs WCDFP Ratio comparison.")
    parser.add_argument("--mode", type=int, default=mode,
                        help="Coloring mode: 0 (no gradient), 1 (utilization), 2 (task count)")
    args = parser.parse_args()
    mode = args.mode
    print(f"Selected mode: {mode}")

    cmap = (LinearSegmentedColormap.from_list("custom_YlGn",
            ["#9ACD32", "#32CD32", "#006400"])
            if mode == 1 else
            LinearSegmentedColormap.from_list("custom_Blues",
            ["#1E90FF", "#4682B4", "#00008B"]))

    fig = plt.figure(figsize=(8 * cols, 6 * rows))
    spec = gridspec.GridSpec(rows, cols, figure=fig, wspace=0.4, hspace=0.2)
    axes = [fig.add_subplot(spec[i]) for i in range(rows * cols)]
    scatter_handles = None
    plot_idx = 0

    # Iterate over unique method pairs
    for i in range(len(METHOD_NAMES)):
        for j in range(i + 1, len(METHOD_NAMES)):
            if plot_idx >= len(axes):
                break
            s = plot_scatter_for_methods(METHOD_NAMES[i], METHOD_NAMES[j], axes[plot_idx], mode, cmap)
            if s is not None:
                scatter_handles = s
            plot_idx += 1

    if mode in [1, 2] and scatter_handles:
        _, __, color_label = get_color_params(mode)
        cbar = fig.colorbar(scatter_handles, ax=axes, aspect=40, shrink=0.8, pad=0.03, location="right")
        if mode == 2:
            ticks = np.linspace(0, 1, 10)
            tick_labels = np.linspace(min(TASK_COUNTS), max(TASK_COUNTS), 10, dtype=int)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(tick_labels)
        cbar.set_label(color_label, rotation=270, labelpad=0, fontsize=16)
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.yaxis.labelpad = 40

    output = os.path.join(OUTPUT_PREFIX, output_file)
    plt.savefig(output, dpi=300, format="pdf", bbox_inches='tight')
    print(f"Merged plot saved as {output_file}")


def plot_wcdfp_comparison(mode=0, rows=2, cols=3, output_file="merged_plot.pdf"):
    """
    Merge scatter plots comparing WCDFP between method pairs into a grid.
    """
    parser = argparse.ArgumentParser(description="Plot WCDFP comparison.")
    parser.add_argument("--mode", type=int, default=mode,
                        help="Coloring mode: 0 (no gradient), 1 (utilization), 2 (task count)")
    args = parser.parse_args()
    mode = args.mode

    cmap = (LinearSegmentedColormap.from_list("custom_YlGn",
            ["#9ACD32", "#32CD32", "#006400"])
            if mode == 1 else
            LinearSegmentedColormap.from_list("custom_Blues",
            ["#1E90FF", "#4682B4", "#00008B"]))

    fig = plt.figure(figsize=(8 * cols, 6 * rows))
    spec = gridspec.GridSpec(rows, cols, figure=fig, wspace=0.4, hspace=0.2)
    axes = [fig.add_subplot(spec[i]) for i in range(rows * cols)]
    scatter_handles = None
    plot_idx = 0
    color_min, color_max, color_label = get_color_params(mode)

    for i in range(len(METHOD_NAMES)):
        for j in range(i + 1, len(METHOD_NAMES)):
            if plot_idx >= len(axes):
                break

            wcdfp_data = []
            color_values = []
            for task_count in TASK_COUNTS:
                for util_rate in UTIL_RATE_VALUES:
                    data_pair = {}
                    for method in [METHOD_NAMES[i], METHOD_NAMES[j]]:
                        prefix = get_prefix(task_count, util_rate)
                        csv_path = os.path.join(prefix, f"evaluation_{method.lower()}.csv")
                        df = load_csv_data(csv_path, columns=["WCDFP"])
                        data_pair[method] = df["WCDFP"].values if df is not None else None
                    if data_pair[METHOD_NAMES[i]] is None or data_pair[METHOD_NAMES[j]] is None:
                        continue
                    min_len = min(len(data_pair[METHOD_NAMES[i]]), len(data_pair[METHOD_NAMES[j]]))
                    wcdfp_data.extend(
                        zip(data_pair[METHOD_NAMES[i]][:min_len], data_pair[METHOD_NAMES[j]][:min_len])
                    )
                    if mode in [1, 2]:
                        color_key = util_rate if mode == 1 else task_count
                        norm_color = (color_key - color_min) / (color_max - color_min)
                        color_values.extend([norm_color] * min_len)

            if not wcdfp_data:
                continue

            x_vals = [v[0] for v in wcdfp_data]
            y_vals = [v[1] for v in wcdfp_data]
            sc = axes[plot_idx].scatter(
                x_vals, y_vals, c=color_values if mode in [1, 2] else "orange",
                s=5, cmap=cmap, vmin=0, vmax=1
            )
            scatter_handles = sc

            # Count points above and below y = x
            above = sum(1 for x, y in zip(x_vals, y_vals) if y > x)
            below = sum(1 for x, y in zip(x_vals, y_vals) if y <= x)
            axes[plot_idx].text(0.02, 0.98, f"{above}",
                                transform=axes[plot_idx].transAxes, fontsize=12,
                                ha='left', va='top', bbox=dict(facecolor='white', edgecolor='black', boxstyle="round,pad=0.3"))
            axes[plot_idx].text(0.98, 0.02, f"{below}",
                                transform=axes[plot_idx].transAxes, fontsize=12,
                                ha='right', va='bottom', bbox=dict(facecolor='white', edgecolor='black', boxstyle="round,pad=0.3"))
            
            min_val = min(min(x_vals), min(y_vals))
            max_val = max(max(x_vals), max(y_vals))
            axes[plot_idx].plot([min_val, max_val], [min_val, max_val], linestyle="--", color="#0F4D48", zorder=10)
            axes[plot_idx].set_xscale("log")
            axes[plot_idx].set_yscale("log")
            axes[plot_idx].set_xlabel(f"WCDFP by $\\bf{{{METHOD_TO_LABEL[METHOD_NAMES[i]]}}}$", fontsize=16)
            axes[plot_idx].set_ylabel(f"WCDFP by $\\bf{{{METHOD_TO_LABEL[METHOD_NAMES[j]]}}}$", fontsize=16)
            axes[plot_idx].tick_params(axis='both', which='major', labelsize=14)
            axes[plot_idx].grid(visible=True, which='major', linestyle='--', linewidth=0.5)
            axes[plot_idx].set_box_aspect(1)
            plot_idx += 1

    if mode in [1, 2] and scatter_handles:
        cbar = fig.colorbar(scatter_handles, ax=axes, aspect=40, shrink=0.8, pad=0.03, location="right")
        if mode == 2:
            ticks = np.linspace(0, 1, 10)
            tick_labels = np.linspace(min(TASK_COUNTS), max(TASK_COUNTS), 10, dtype=int)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(tick_labels)
        cbar.set_label(color_label, rotation=270, labelpad=10, fontsize=16)
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.yaxis.labelpad = 40

    output = os.path.join(OUTPUT_PREFIX, output_file)
    plt.savefig(output, dpi=300, format="pdf", bbox_inches='tight')
    print(f"Merged plot saved as {output_file}")


def plot_comparison_for_task_id(task_id=35, task_count=50, util_rate=0.60):
    """
    Plots Execution Time vs WCDFP for a specific task ID across methods.
    """
    method_colors = {
        "monte_carlo_multi": "tab:blue",
        "monte_carlo_single": "tab:purple",
        "berry_essen": "tab:orange",
        "sequential_conv": "tab:green",
        "aggregate_conv_imp": "tab:red"
    }
    methods = ["monte_carlo_multi", "monte_carlo_single", "berry_essen", "sequential_conv", "aggregate_conv_imp"]
    plt.figure(figsize=(12, 7))
    for method in methods:
        suffix = f"_{task_id}" if "monte_carlo" in method else ""
        input_file = f"evaluation_{method}{suffix}.csv"
        prefix = os.path.join(OUTPUT_PREFIX, f"{task_count}_{util_rate:.2f}_{EPSILON}")
        file_path = os.path.join(prefix, input_file)
        df = load_csv_data(file_path)
        if df is None:
            continue
        df = df[df["TaskSetID"] == task_id]
        x, y = df["ExecutionTime"], df["WCDFP"]
        label = METHOD_TO_LABEL.get(method, method)
        if method.startswith("monte_carlo"):
            plt.plot(x, y, marker='o', linestyle='-', color=method_colors[method],
                     label=label, linewidth=1.5)
        else:
            plt.scatter(x, y, color=method_colors[method], s=60, label=label)
            for val in y:
                plt.axhline(y=val, xmin=0, xmax=1, color=method_colors[method],
                            linestyle='--', linewidth=1.5, alpha=0.7)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Execution Time [s]", fontsize=20, labelpad=16)
    plt.ylabel("WCDFP", fontsize=20, labelpad=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.minorticks_off()
    plt.legend(fontsize=16, loc='upper right')
    plt.tight_layout()
    output = os.path.join(OUTPUT_PREFIX, f"comparison_taskset_{task_id}.pdf")
    plt.savefig(output, dpi=300, format="pdf", bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output}")
