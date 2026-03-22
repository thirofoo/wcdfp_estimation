import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.transforms as mtransforms
from matplotlib.colors import LinearSegmentedColormap
import fitz

# Global configuration
OUTPUT_PREFIX = "src/evaluation/output"
PREFIX_TEMPLATE = os.path.join("src", "evaluation", "output", "{task_count}_{util_rate:.2f}_{epsilon}")

BASE_METHOD_NAMES = [
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
RESCALED_METHOD_PREFIX = "aggregate_conv_imp_rescaled_lmax"

# Constant parameter values for most plots
TASK_COUNTS = range(10, 101, 10)
UTIL_RATE_VALUES = [0.6, 0.65, 0.70]
EPSILON = "0.001"


def get_prefix(task_count, util_rate):
    return PREFIX_TEMPLATE.format(task_count=task_count, util_rate=util_rate, epsilon=EPSILON)


def parse_rescaled_lmax(method_name):
    if not method_name.startswith(RESCALED_METHOD_PREFIX):
        return None
    suffix = method_name[len(RESCALED_METHOD_PREFIX):]
    return int(suffix) if suffix.isdigit() else None


def discover_rescaled_methods(min_acr_lmax=None, max_acr_lmax=None):
    discovered = set()
    for task_count in TASK_COUNTS:
        for util_rate in UTIL_RATE_VALUES:
            prefix = get_prefix(task_count, util_rate)
            if not os.path.isdir(prefix):
                continue
            for file_name in os.listdir(prefix):
                if not (file_name.startswith("evaluation_") and file_name.endswith(".csv")):
                    continue
                method_name = file_name[len("evaluation_"):-len(".csv")]
                if method_name.startswith(RESCALED_METHOD_PREFIX):
                    lmax = parse_rescaled_lmax(method_name)
                    if lmax is None:
                        continue
                    if min_acr_lmax is not None and lmax < int(min_acr_lmax):
                        continue
                    if max_acr_lmax is not None and lmax > int(max_acr_lmax):
                        continue
                    discovered.add(method_name)

    return sorted(discovered, key=lambda name: parse_rescaled_lmax(name) or 0)


def get_method_names(min_acr_lmax=None, max_acr_lmax=None):
    return BASE_METHOD_NAMES + discover_rescaled_methods(
        min_acr_lmax=min_acr_lmax,
        max_acr_lmax=max_acr_lmax,
    )


def get_method_pairs(ac_vs_acr_only=False, min_acr_lmax=None, max_acr_lmax=None):
    method_names = get_method_names(
        min_acr_lmax=min_acr_lmax,
        max_acr_lmax=max_acr_lmax,
    )
    if ac_vs_acr_only:
        ac_method = "aggregate_conv_imp"
        if ac_method not in method_names:
            return []
        rescaled_methods = [name for name in method_names if parse_rescaled_lmax(name) is not None]
        return [(ac_method, method) for method in rescaled_methods]
    return [
        (method_names[i], method_names[j])
        for i in range(len(method_names))
        for j in range(i + 1, len(method_names))
    ]


def get_method_label(method_name):
    if method_name in METHOD_TO_LABEL:
        return METHOD_TO_LABEL[method_name]
    lmax = parse_rescaled_lmax(method_name)
    if lmax is not None:
        return f"ACR({lmax})"
    return method_name


def load_csv_data(file_path, columns=None, verbose=False):
    """
    Returns a DataFrame for a given file path. Optionally checks for required columns.
    """
    if not os.path.exists(file_path):
        if verbose:
            print(f"File not found: {file_path}")
        return None
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        if verbose:
            print(f"Error reading {file_path}: {e}")
        return None

    if columns:
        for col in columns:
            if col not in df.columns:
                if verbose:
                    print(f"The file {file_path} does not contain '{col}' column.")
                return None
    return df


def plot_execution_time_boxplot(min_acr_lmax=None, max_acr_lmax=None):
    """
    Aggregates execution times across parameter values and produces a horizontal log-scale boxplot.
    """
    parser = argparse.ArgumentParser(description="Plot execution-time boxplot.")
    parser.add_argument(
        "--min-acr-lmax",
        type=int,
        default=min_acr_lmax,
        help="Exclude ACR methods with Lmax smaller than this value",
    )
    parser.add_argument(
        "--max-acr-lmax",
        type=int,
        default=max_acr_lmax,
        help="Exclude ACR methods with Lmax greater than this value",
    )
    args = parser.parse_args()
    min_acr_lmax = args.min_acr_lmax
    max_acr_lmax = args.max_acr_lmax

    all_methods_times = []
    active_methods = []
    for method in get_method_names(min_acr_lmax=min_acr_lmax, max_acr_lmax=max_acr_lmax):
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
        if method_times:
            all_methods_times.append(method_times)
            active_methods.append(method)

    if not all_methods_times:
        print("No execution-time data found for plotting.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(all_methods_times, patch_artist=True, whis=[0, 100],
               showfliers=True, vert=False, widths=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Execution Time [s]", fontsize=20, labelpad=16)
    ax.set_ylim(0.5, len(active_methods) + 0.5)
    ax.margins(y=0.0)
    plt.rc('text', usetex=False)
    plt.rc('font', family='serif')
    ax.set_yticks(range(1, len(active_methods) + 1))
    ax.set_yticklabels([f"$\\mathbf{{{get_method_label(m)}}}$" for m in active_methods],
                       va="center", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(axis="x", linestyle="--", linewidth=0.5)

    # Show average execution time next to each box for easier reading on log scale.
    avg_times = [float(np.mean(method_times)) for method_times in all_methods_times]
    text_transform = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    for idx, avg_time in enumerate(avg_times, start=1):
        ax.text(
            1.01,
            idx,
            f"avg={avg_time:.3e}s",
            transform=text_transform,
            va="center",
            ha="left",
            fontsize=11,
            clip_on=False,
        )
    plt.tight_layout(rect=[0, 0, 0.86, 1])

    output_path = os.path.join(OUTPUT_PREFIX, "execution_time_boxplot_aggregated.pdf")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Aggregated horizontal boxplot saved to {output_path}")


def plot_precision_ratio_boxplot_acr_vs_ac(min_acr_lmax=None, max_acr_lmax=None):
    """
    Aggregates WCDFP ratios ACR/AC across parameter values
    and produces a horizontal log-scale boxplot.
    """
    parser = argparse.ArgumentParser(description="Plot ACR/AC precision ratio boxplot.")
    parser.add_argument(
        "--min-acr-lmax",
        type=int,
        default=min_acr_lmax,
        help="Exclude ACR methods with Lmax smaller than this value",
    )
    parser.add_argument(
        "--max-acr-lmax",
        type=int,
        default=max_acr_lmax,
        help="Exclude ACR methods with Lmax greater than this value",
    )
    args = parser.parse_args()
    min_acr_lmax = args.min_acr_lmax
    max_acr_lmax = args.max_acr_lmax

    ac_method = "aggregate_conv_imp"
    acr_methods = discover_rescaled_methods(min_acr_lmax=min_acr_lmax, max_acr_lmax=max_acr_lmax)
    all_precision_ratios = []
    active_acr_methods = []

    for acr_method in acr_methods:
        ratio_values = []
        for task_count in TASK_COUNTS:
            for util_rate in UTIL_RATE_VALUES:
                prefix = get_prefix(task_count, util_rate)
                ac_path = os.path.join(prefix, f"evaluation_{ac_method}.csv")
                acr_path = os.path.join(prefix, f"evaluation_{acr_method}.csv")
                ac_df = load_csv_data(ac_path, columns=["TaskSetID", "WCDFP"])
                acr_df = load_csv_data(acr_path, columns=["TaskSetID", "WCDFP"])
                if ac_df is None or acr_df is None:
                    continue

                ac_df = ac_df[["TaskSetID", "WCDFP"]].rename(columns={"WCDFP": "WCDFPAC"})
                acr_df = acr_df[["TaskSetID", "WCDFP"]].rename(columns={"WCDFP": "WCDFPACR"})
                merged = pd.merge(ac_df, acr_df, on="TaskSetID", how="inner")
                if merged.empty:
                    continue

                valid = merged[
                    (merged["WCDFPAC"] > 0) &
                    (merged["WCDFPACR"] > 0)
                ]
                if valid.empty:
                    continue
                ratio_values.extend((valid["WCDFPACR"] / valid["WCDFPAC"]).tolist())

        if ratio_values:
            all_precision_ratios.append(ratio_values)
            active_acr_methods.append(acr_method)

    if not all_precision_ratios:
        print("No ACR/AC precision ratio data found for plotting.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(all_precision_ratios, patch_artist=True, whis=[0, 100],
               showfliers=True, vert=False, widths=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("WCDFP Ratio [ACR / AC]", fontsize=20, labelpad=16)
    ax.axvline(x=1.0, linestyle="--", color="#0F4D48", linewidth=1.0, zorder=10)
    ax.set_ylim(0.5, len(active_acr_methods) + 0.5)
    ax.margins(y=0.0)
    plt.rc('text', usetex=False)
    plt.rc('font', family='serif')
    ax.set_yticks(range(1, len(active_acr_methods) + 1))
    ax.set_yticklabels([f"$\\mathbf{{{get_method_label(m)}}}$" for m in active_acr_methods],
                       va="center", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(axis="x", linestyle="--", linewidth=0.5)

    median_ratios = [float(np.median(values)) for values in all_precision_ratios]
    text_transform = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    for idx, median_ratio in enumerate(median_ratios, start=1):
        ax.text(
            1.01,
            idx,
            f"median={median_ratio:.3e}",
            transform=text_transform,
            va="center",
            ha="left",
            fontsize=11,
            clip_on=False,
        )
    plt.tight_layout(rect=[0, 0, 0.86, 1])

    output_path = os.path.join(OUTPUT_PREFIX, "precision_ratio_boxplot_acr_vs_ac.pdf")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"ACR/AC precision ratio horizontal boxplot saved to {output_path}")


def plot_execution_time_ratio_boxplot_ac_vs_acr(min_acr_lmax=None, max_acr_lmax=None):
    """
    Backward-compatible alias. The plotted metric is now WCDFP ratio (ACR/AC).
    """
    return plot_precision_ratio_boxplot_acr_vs_ac(
        min_acr_lmax=min_acr_lmax,
        max_acr_lmax=max_acr_lmax,
    )


def plot_execution_time_vs_precision_ratio_acr_vs_ac(min_acr_lmax=None, max_acr_lmax=None):
    """
    Plots a tradeoff curve where each point corresponds to one ACR(Lmax):
    x = median execution time of ACR, y = median WCDFP ratio (ACR/AC).
    """
    parser = argparse.ArgumentParser(description="Plot execution-time vs precision-ratio tradeoff for ACR.")
    parser.add_argument(
        "--min-acr-lmax",
        type=int,
        default=min_acr_lmax,
        help="Exclude ACR methods with Lmax smaller than this value",
    )
    parser.add_argument(
        "--max-acr-lmax",
        type=int,
        default=max_acr_lmax,
        help="Exclude ACR methods with Lmax greater than this value",
    )
    args = parser.parse_args()
    min_acr_lmax = args.min_acr_lmax
    max_acr_lmax = args.max_acr_lmax

    ac_method = "aggregate_conv_imp"
    acr_methods = discover_rescaled_methods(min_acr_lmax=min_acr_lmax, max_acr_lmax=max_acr_lmax)

    points = []
    for acr_method in acr_methods:
        exec_times = []
        precision_ratios = []
        for task_count in TASK_COUNTS:
            for util_rate in UTIL_RATE_VALUES:
                prefix = get_prefix(task_count, util_rate)
                ac_path = os.path.join(prefix, f"evaluation_{ac_method}.csv")
                acr_path = os.path.join(prefix, f"evaluation_{acr_method}.csv")
                ac_df = load_csv_data(ac_path, columns=["TaskSetID", "ExecutionTime", "WCDFP"])
                acr_df = load_csv_data(acr_path, columns=["TaskSetID", "ExecutionTime", "WCDFP"])
                if ac_df is None or acr_df is None:
                    continue

                ac_df = ac_df.rename(
                    columns={"ExecutionTime": "ExecutionTimeAC", "WCDFP": "WCDFPAC"}
                )[["TaskSetID", "ExecutionTimeAC", "WCDFPAC"]]
                acr_df = acr_df.rename(
                    columns={"ExecutionTime": "ExecutionTimeACR", "WCDFP": "WCDFPACR"}
                )[["TaskSetID", "ExecutionTimeACR", "WCDFPACR"]]
                merged = pd.merge(ac_df, acr_df, on="TaskSetID", how="inner")
                if merged.empty:
                    continue

                valid = merged[
                    (merged["ExecutionTimeACR"] >= 0) &
                    (merged["WCDFPAC"] > 0) &
                    (merged["WCDFPACR"] > 0)
                ]
                if valid.empty:
                    continue

                exec_times.extend(valid["ExecutionTimeACR"].tolist())
                precision_ratios.extend((valid["WCDFPACR"] / valid["WCDFPAC"]).tolist())

        if exec_times and precision_ratios:
            points.append(
                (
                    parse_rescaled_lmax(acr_method),
                    float(np.median(exec_times)),
                    float(np.median(precision_ratios)),
                    acr_method,
                )
            )

    if not points:
        print("No ACR tradeoff data found for plotting.")
        return

    points.sort(key=lambda item: item[0] if item[0] is not None else 0)
    x_vals = [p[1] for p in points]
    y_vals = [p[2] for p in points]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_vals, y_vals, marker="o", linewidth=1.8, color="#1E90FF")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(1.0, linestyle="--", color="#0F4D48", linewidth=1.0, zorder=10)
    ax.set_xlabel("Median Execution Time [s] (ACR)", fontsize=20, labelpad=16)
    ax.set_ylabel("Median WCDFP Ratio [ACR / AC]", fontsize=20, labelpad=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(True, which="major", linestyle="--", linewidth=0.5)

    for lmax, x, y, _ in points:
        ax.annotate(
            f"L={lmax}",
            xy=(x, y),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=10,
        )

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_PREFIX, "execution_time_vs_precision_ratio_acr_vs_ac.pdf")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"ACR tradeoff plot saved to {output_path}")


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
    if mode in [1, 2]:
        scatter = ax.scatter(x_vals, y_vals, c=color_values, cmap=custom_cmap, vmin=0, vmax=1, s=5)
    else:
        scatter = ax.scatter(x_vals, y_vals, c="orange", s=5)
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
    label_1 = get_method_label(method_1)
    label_2 = get_method_label(method_2)
    ax.set_xlabel(f"Execution Time Ratio ($\\bf{{{label_1}}} / \\bf{{{label_2}}}$)", fontsize=16)
    ax.set_ylabel(f"WCDFP Ratio ($\\bf{{{label_1}}} / \\bf{{{label_2}}}$)", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(visible=True, which='major', linestyle='--', linewidth=0.5)
    ax.set_box_aspect(1)
    return scatter


def plot_time_ratio_vs_wcdfp_ratio(mode=0, rows=None, cols=3, output_file="merged_ratio_plot.pdf"):
    """
    Merge Time Ratio vs WCDFP Ratio scatter plots into a grid layout.
    Mode 0: no color gradient, 1: gradient by utilization, 2: gradient by task count.
    """
    parser = argparse.ArgumentParser(description="Plot Time Ratio vs WCDFP Ratio comparison.")
    parser.add_argument("--mode", type=int, default=mode,
                        help="Coloring mode: 0 (no gradient), 1 (utilization), 2 (task count)")
    parser.add_argument(
        "--ac-vs-acr-only",
        action="store_true",
        help="Plot only AC vs each ACR(Lmax) pair",
    )
    parser.add_argument(
        "--min-acr-lmax",
        type=int,
        default=None,
        help="Exclude ACR methods with Lmax smaller than this value",
    )
    parser.add_argument(
        "--max-acr-lmax",
        type=int,
        default=None,
        help="Exclude ACR methods with Lmax greater than this value",
    )
    args = parser.parse_args()
    mode = args.mode
    ac_vs_acr_only = args.ac_vs_acr_only
    min_acr_lmax = args.min_acr_lmax
    max_acr_lmax = args.max_acr_lmax
    print(f"Selected mode: {mode}")

    cmap = (LinearSegmentedColormap.from_list("custom_YlGn",
            ["#9ACD32", "#32CD32", "#006400"])
            if mode == 1 else
            LinearSegmentedColormap.from_list("custom_Blues",
            ["#1E90FF", "#4682B4", "#00008B"]))

    method_pairs = get_method_pairs(
        ac_vs_acr_only=ac_vs_acr_only,
        min_acr_lmax=min_acr_lmax,
        max_acr_lmax=max_acr_lmax,
    )
    pair_count = len(method_pairs)
    if pair_count == 0:
        print("No method pair available for plotting.")
        return
    if rows is None:
        rows = int(np.ceil(pair_count / cols))
    elif rows * cols < pair_count:
        rows = int(np.ceil(pair_count / cols))

    fig = plt.figure(figsize=(8 * cols, 6 * rows))
    spec = gridspec.GridSpec(rows, cols, figure=fig, wspace=0.4, hspace=0.2)
    axes = [fig.add_subplot(spec[i]) for i in range(rows * cols)]
    scatter_handles = None
    plot_idx = 0

    # Iterate over unique method pairs
    for method_1, method_2 in method_pairs:
        if plot_idx >= len(axes):
            break
        s = plot_scatter_for_methods(method_1, method_2, axes[plot_idx], mode, cmap)
        if s is not None:
            scatter_handles = s
        plot_idx += 1
    for ax in axes[plot_idx:]:
        ax.axis("off")

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


def plot_wcdfp_comparison(mode=0, rows=None, cols=3, output_file="merged_plot.pdf"):
    """
    Merge scatter plots comparing WCDFP between method pairs into a grid.
    """
    parser = argparse.ArgumentParser(description="Plot WCDFP comparison.")
    parser.add_argument("--mode", type=int, default=mode,
                        help="Coloring mode: 0 (no gradient), 1 (utilization), 2 (task count)")
    parser.add_argument(
        "--ac-vs-acr-only",
        action="store_true",
        help="Plot only AC vs each ACR(Lmax) pair",
    )
    parser.add_argument(
        "--min-acr-lmax",
        type=int,
        default=None,
        help="Exclude ACR methods with Lmax smaller than this value",
    )
    parser.add_argument(
        "--max-acr-lmax",
        type=int,
        default=None,
        help="Exclude ACR methods with Lmax greater than this value",
    )
    args = parser.parse_args()
    mode = args.mode
    ac_vs_acr_only = args.ac_vs_acr_only
    min_acr_lmax = args.min_acr_lmax
    max_acr_lmax = args.max_acr_lmax

    cmap = (LinearSegmentedColormap.from_list("custom_YlGn",
            ["#9ACD32", "#32CD32", "#006400"])
            if mode == 1 else
            LinearSegmentedColormap.from_list("custom_Blues",
            ["#1E90FF", "#4682B4", "#00008B"]))

    method_pairs = get_method_pairs(
        ac_vs_acr_only=ac_vs_acr_only,
        min_acr_lmax=min_acr_lmax,
        max_acr_lmax=max_acr_lmax,
    )
    pair_count = len(method_pairs)
    if pair_count == 0:
        print("No method pair available for plotting.")
        return
    if rows is None:
        rows = int(np.ceil(pair_count / cols))
    elif rows * cols < pair_count:
        rows = int(np.ceil(pair_count / cols))

    fig = plt.figure(figsize=(8 * cols, 6 * rows))
    spec = gridspec.GridSpec(rows, cols, figure=fig, wspace=0.4, hspace=0.2)
    axes = [fig.add_subplot(spec[i]) for i in range(rows * cols)]
    scatter_handles = None
    plot_idx = 0
    color_min, color_max, color_label = get_color_params(mode)

    for method_1, method_2 in method_pairs:
        if plot_idx >= len(axes):
            break

        wcdfp_data = []
        color_values = []
        for task_count in TASK_COUNTS:
            for util_rate in UTIL_RATE_VALUES:
                data_pair = {}
                for method in [method_1, method_2]:
                    prefix = get_prefix(task_count, util_rate)
                    csv_path = os.path.join(prefix, f"evaluation_{method.lower()}.csv")
                    df = load_csv_data(csv_path, columns=["WCDFP"])
                    data_pair[method] = df["WCDFP"].values if df is not None else None
                if data_pair[method_1] is None or data_pair[method_2] is None:
                    continue
                min_len = min(len(data_pair[method_1]), len(data_pair[method_2]))
                wcdfp_data.extend(
                    zip(data_pair[method_1][:min_len], data_pair[method_2][:min_len])
                )
                if mode in [1, 2]:
                    color_key = util_rate if mode == 1 else task_count
                    norm_color = (color_key - color_min) / (color_max - color_min)
                    color_values.extend([norm_color] * min_len)

        if not wcdfp_data:
            continue

        x_vals = [v[0] for v in wcdfp_data]
        y_vals = [v[1] for v in wcdfp_data]
        if mode in [1, 2]:
            sc = axes[plot_idx].scatter(x_vals, y_vals, c=color_values, s=5, cmap=cmap, vmin=0, vmax=1)
        else:
            sc = axes[plot_idx].scatter(x_vals, y_vals, c="orange", s=5)
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
        label_1 = get_method_label(method_1)
        label_2 = get_method_label(method_2)
        axes[plot_idx].set_xlabel(f"WCDFP by $\\bf{{{label_1}}}$", fontsize=16)
        axes[plot_idx].set_ylabel(f"WCDFP by $\\bf{{{label_2}}}$", fontsize=16)
        axes[plot_idx].tick_params(axis='both', which='major', labelsize=14)
        axes[plot_idx].grid(visible=True, which='major', linestyle='--', linewidth=0.5)
        axes[plot_idx].set_box_aspect(1)
        plot_idx += 1
    for ax in axes[plot_idx:]:
        ax.axis("off")

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
    methods = get_method_names()
    method_colors = plt.cm.tab10(np.linspace(0.0, 1.0, len(methods)))
    plt.figure(figsize=(12, 7))
    for idx, method in enumerate(methods):
        input_file = f"evaluation_{method}.csv"
        prefix = os.path.join(OUTPUT_PREFIX, f"{task_count}_{util_rate:.2f}_{EPSILON}")
        file_path = os.path.join(prefix, input_file)
        df = load_csv_data(file_path)
        if df is None:
            continue
        df = df[df["TaskSetID"] == task_id]
        if df.empty:
            continue
        x, y = df["ExecutionTime"], df["WCDFP"]
        label = get_method_label(method)
        plt.scatter(x, y, color=method_colors[idx], s=60, label=label)
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
