import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import argparse
import fitz
import os

# output_prefix = "src/evaluation/output"
output_prefix = "src/evaluation/output_0117"
# prefix_template = "src/evaluation/output/{task_count}_{util_rate:.2f}_{epsilon}"
prefix_template = "src/evaluation/output_0117/{task_count}_{util_rate:.2f}_{epsilon}"

method_names = [
    "berry_essen",
    "convolution_doubling",
    "convolution_merge",
    # "monte_carlo",
    "convolution",
    # "convolution_doubling_float128",
    # "convolution_float128"
]
method_to_label = {
    "berry_essen": "BE",
    # "convolution_doubling": "AC \ (Orig.)",
    # "convolution_merge": "AC \ (Imp.)",
    "convolution_merge": "",
    "convolution_doubling": "",
    "convolution": "SC",
    "monte_carlo": "MC",
    "monte_carlo_multi": "MC (16 thread)",
    "monte_carlo_single": "MC (1 thread)",
    # "convolution_doubling_float128": "RC (float128)",
    # "convolution_float128": "SC (float128)"
}

def plot_execution_time_boxplot():
    """
    Consider all combinations of (task_count, util_rate, epsilon),
    aggregate the execution times for each method into one list,
    and create a single horizontal boxplot figure where each method corresponds to one box.
    """

    # Parameter settings
    task_counts = range(10, 101, 10)  # 10, 20, ..., 100
    util_rate_values = [0.6, 0.65, 0.7]
    epsilon = "0.001"

    # A list to store aggregated execution times for each method
    all_methods_execution_times = []

    # Aggregate execution times for all parameter combinations per method
    for method_name in method_names:
        aggregated_times_for_method = []

        for task_count in task_counts:
            for util_rate in util_rate_values:
                # Construct file path (assuming prefix_template is defined elsewhere)
                prefix = prefix_template.format(task_count=task_count, util_rate=util_rate, epsilon=epsilon)
                csv_path = os.path.join(prefix, f"evaluation_{method_name.lower()}.csv")

                if not os.path.exists(csv_path):
                    print(f"File not found: {csv_path}")
                    continue

                data = pd.read_csv(csv_path)
                if "ExecutionTime" not in data.columns:
                    print(f"The file {csv_path} does not contain an 'ExecutionTime' column.")
                    continue

                # Filter execution times (non-negative) and append to the list
                filtered_times = data["ExecutionTime"][data["ExecutionTime"] >= 0].tolist()
                aggregated_times_for_method.extend(filtered_times)

        # Append the data for one method to the overall list
        all_methods_execution_times.append(aggregated_times_for_method)

    # Create the horizontal boxplot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Horizontal boxplot configuration
    boxplot_data = ax.boxplot(
        all_methods_execution_times,
        patch_artist=True,    # Allow filled boxes
        whis=[0, 100],        # Whiskers at min and max
        showfliers=True,      # Show outliers
        vert=False            # Make the boxplot horizontal
    )

    # Use a logarithmic scale on the x-axis
    ax.set_xscale("log")
    ax.set_xlabel("Execution Time (log scale)", fontsize=16)

    # Set y-axis labels (method names)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    ax.set_yticks(range(1, len(method_names) + 1))
    ax.set_yticklabels([f"$\\mathbf{{{method_to_label[m]}}}$" for m in method_names], va="center", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)

    # Add grid on the x-axis
    ax.grid(axis="x", linestyle="--", linewidth=0.5)

    # Adjust layout and save the figure as PDF
    plt.tight_layout()
    output_path = os.path.join(output_prefix, "execution_time_boxplot_aggregated.pdf")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Aggregated horizontal boxplot saved to {output_path}")




def plot_time_ratio_vs_wcdfp_ratio(mode=0, rows=3, cols=2, output_file="merged_ratio_plot.pdf"):
    """
    Merge all Time Ratio vs WCDFP Ratio plots into a single grid layout with a shared colorbar.
    Each plot follows the original design, centered around (1, 1) with reference lines and margins.

    :param mode: Coloring mode for the plot.
                 0 - No color gradient.
                 1 - Gradient based on utilization rate.
                 2 - Gradient based on task count.
    :param rows: Number of rows in the grid.
    :param cols: Number of columns in the grid.
    :param output_file: Output file name for the merged plot.
    """
    task_counts = range(10, 101, 10)
    util_rate_values = [0.6, 0.65, 0.70]
    epsilon = "0.001"

    # Parse arguments
    parser = argparse.ArgumentParser(description="Plot Time Ratio vs WCDFP Ratio comparison.")
    parser.add_argument("--mode", type=int, default=0, help="Coloring mode: 0 (no gradient), 1 (utilization), 2 (task count)")
    args = parser.parse_args()
    mode = args.mode
    print(f"Selected mode: {mode}")

    # Custom colormaps
    custom_YlGn = LinearSegmentedColormap.from_list("custom_YlGn", ["#9ACD32", "#32CD32", "#006400"])
    custom_Blues = LinearSegmentedColormap.from_list("custom_Blues", ["#1E90FF", "#4682B4", "#00008B"])

    fig = plt.figure(figsize=(7 * cols, 6 * rows))
    spec = gridspec.GridSpec(rows, cols, figure=fig, wspace=0.4, hspace=0.1)
    axes = [fig.add_subplot(spec[row, col]) for row in range(rows) for col in range(cols)]

    plot_idx = 0
    scatter_handles = None

    for i in range(len(method_names)):
        for j in range(i + 1, len(method_names)):
            if plot_idx >= len(axes):
                break

            method_1 = method_names[i]
            method_2 = method_names[j]

            ratio_data = []
            color_values = []

            # Gradient calculation ranges
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
                    metrics = {}
                    for method in [method_1, method_2]:
                        prefix = prefix_template.format(task_count=task_count, util_rate=util_rate, epsilon=epsilon)
                        csv_path = os.path.join(prefix, f"evaluation_{method.lower()}.csv")
                        try:
                            data = pd.read_csv(csv_path)
                            metrics[method] = {
                                "WCDFP": data["WCDFP"].values,
                                "ExecutionTime": data["ExecutionTime"].values
                            }
                        except FileNotFoundError:
                            metrics[method] = None

                    if metrics[method_1] and metrics[method_2]:
                        wcdfp_1 = metrics[method_1]["WCDFP"]
                        wcdfp_2 = metrics[method_2]["WCDFP"]
                        time_1 = metrics[method_1]["ExecutionTime"]
                        time_2 = metrics[method_2]["ExecutionTime"]

                        min_len = min(len(wcdfp_1), len(wcdfp_2))
                        for k in range(min_len):
                            time_ratio = time_1[k] / time_2[k]
                            wcdfp_ratio = wcdfp_1[k] / wcdfp_2[k]
                            ratio_data.append((time_ratio, wcdfp_ratio))

                            # if time_ratio < 1:
                            #     print(f"Time ratio is less than 1: {time_ratio} ({method_1}: {time_1[k]}, {method_2}: {time_2[k]})")
                            #     print(f"task_count: {task_count}, util_rate: {util_rate}, epsilon: {epsilon}")

                            if mode in [1, 2]:
                                color_key = util_rate if mode == 1 else task_count
                                normalized_color = (color_key - color_min) / (color_max - color_min)
                                color_values.append(normalized_color)

            if not ratio_data:
                continue

            x_vals = [x[0] for x in ratio_data]
            y_vals = [x[1] for x in ratio_data]
            ax = axes[plot_idx]

            # Scatter plot
            cmap = custom_YlGn if mode == 1 else custom_Blues
            scatter_handles = ax.scatter(x_vals, y_vals, c=color_values if mode in [1, 2] else "orange", cmap=cmap, vmin=0, vmax=1, s=5)

            # Reference lines
            ax.axvline(x=1, linestyle="--", color="#0F4D48", zorder=10)
            ax.axhline(y=1, linestyle="--", color="#0F4D48", zorder=10)

            # Adjust limits so (1, 1) is at the center
            margin_lower = 0.1
            margin_upper = 1.0 / margin_lower
            min_x, min_y = min(x_vals) * margin_lower, min(y_vals) * margin_lower
            max_x, max_y = max(x_vals) * margin_upper, max(y_vals) * margin_upper
            using_x_abs = max(1 / min_x, max_x)
            using_y_abs = max(1 / min_y, max_y)

            # Count points in each quadrant
            quad_counts = {
                "Q1": sum(1 for x, y in zip(x_vals, y_vals) if x > 1 and y > 1),
                "Q2": sum(1 for x, y in zip(x_vals, y_vals) if x <= 1 and y > 1),
                "Q3": sum(1 for x, y in zip(x_vals, y_vals) if x <= 1 and y <= 1),
                "Q4": sum(1 for x, y in zip(x_vals, y_vals) if x > 1 and y <= 1),
            }

            # Add labels at the corners of the plot
            ax.text(0.02, 0.98, f"{quad_counts['Q2']}", 
                    transform=ax.transAxes, fontsize=12, 
                    ha='left', va='top', bbox=dict(facecolor='white', edgecolor='black', boxstyle="round,pad=0.3"))
            ax.text(0.98, 0.98, f"{quad_counts['Q1']}", 
                    transform=ax.transAxes, fontsize=12, 
                    ha='right', va='top', bbox=dict(facecolor='white', edgecolor='black', boxstyle="round,pad=0.3"))
            ax.text(0.02, 0.02, f"{quad_counts['Q3']}", 
                    transform=ax.transAxes, fontsize=12, 
                    ha='left', va='bottom', bbox=dict(facecolor='white', edgecolor='black', boxstyle="round,pad=0.3"))
            ax.text(0.98, 0.02, f"{quad_counts['Q4']}", 
                    transform=ax.transAxes, fontsize=12, 
                    ha='right', va='bottom', bbox=dict(facecolor='white', edgecolor='black', boxstyle="round,pad=0.3"))

            ax.set_xlim([1 / using_x_abs / 10, using_x_abs * 10])
            ax.set_ylim([1 / using_y_abs / 10, using_y_abs * 10])

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(f"Execution Time Ratio ($\\bf{{{method_to_label[method_1]}}} / \\bf{{{method_to_label[method_2]}}}$)", fontsize=16)
            ax.set_ylabel(f"WCDFP Ratio ($\\bf{{{method_to_label[method_1]}}} / \\bf{{{method_to_label[method_2]}}}$)", fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=14)

            ax.grid(visible=True, which='major', linestyle='--', linewidth=0.5)
            ax.set_box_aspect(1)

            plot_idx += 1

    # Add colorbar
    if mode in [1, 2]:
        cbar = fig.colorbar(scatter_handles, ax=axes, aspect=50, shrink=0.7, pad=0.04, location="right")
        cbar.set_label(color_label, rotation=270, labelpad=10, fontsize=16)
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.yaxis.labelpad = 40

    # Save merged plot
    output = os.path.join(output_prefix, output_file)
    plt.savefig(output, dpi=300, format="pdf", bbox_inches='tight')
    print(f"Merged plot saved as {output_file}")


def plot_wcdfp_comparison(mode=0, rows=3, cols=2, output_file="merged_plot.pdf"):
    """
    Generate a grid of scatter plots comparing WCDFP values between method pairs.

    :param mode: Coloring mode for the plot.
                 0 - No color gradient.
                 1 - Gradient based on utilization rate.
                 2 - Gradient based on task count.
    :param rows: Number of rows in the grid.
    :param cols: Number of columns in the grid.
    :param output_file: Output file name for the merged plot.
    """
    task_counts = range(10, 101, 10)
    util_rate_values = [0.6, 0.65, 0.70]
    epsilon = "0.001"

    # Argument parser
    parser = argparse.ArgumentParser(description="Plot WCDFP comparison.")
    parser.add_argument("--mode", type=int, default=0, help="Coloring mode: 0 (no gradient), 1 (utilization), 2 (task count)")
    args = parser.parse_args()
    mode = args.mode

    # Define colormaps
    custom_YlGn = LinearSegmentedColormap.from_list("custom_YlGn", ["#9ACD32", "#32CD32", "#006400"])
    custom_Blues = LinearSegmentedColormap.from_list("custom_Blues", ["#1E90FF", "#4682B4", "#00008B"])

    fig = plt.figure(figsize=(7 * cols, 6 * rows))
    spec = gridspec.GridSpec(rows, cols, figure=fig, wspace=0.4, hspace=0.1)
    axes = [fig.add_subplot(spec[row, col]) for row in range(rows) for col in range(cols)]

    plot_idx = 0
    scatter_handles = None

    for i in range(len(method_names)):
        for j in range(i + 1, len(method_names)):
            if plot_idx >= len(axes):
                break

            method_1 = method_names[i]
            method_2 = method_names[j]

            wcdfp_data = []
            color_values = []

            # Define gradient bounds
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
                    data_per_method = {}
                    for method_name in [method_1, method_2]:
                        prefix = prefix_template.format(task_count=task_count, util_rate=util_rate, epsilon=epsilon)
                        csv_path = f"{prefix}/evaluation_{method_name.lower()}.csv"
                        try:
                            data = pd.read_csv(csv_path)
                            if "WCDFP" not in data.columns:
                                raise ValueError(f"The file {csv_path} does not contain a 'WCDFP' column.")
                            data_per_method[method_name] = data["WCDFP"].values
                        except FileNotFoundError:
                            print(f"File not found: {csv_path}")
                            data_per_method[method_name] = None

                    wcdfp_1 = data_per_method.get(method_1)
                    wcdfp_2 = data_per_method.get(method_2)

                    if wcdfp_1 is not None and wcdfp_2 is not None:
                        min_len = min(len(wcdfp_1), len(wcdfp_2))
                        wcdfp_data.extend(zip(wcdfp_1[:min_len], wcdfp_2[:min_len]))

                        if mode in [1, 2]:
                            color_key = util_rate if mode == 1 else task_count
                            normalized_color = (color_key - color_min) / (color_max - color_min)
                            color_values.extend([normalized_color] * min_len)

            if not wcdfp_data:
                continue

            x_vals = [x[0] for x in wcdfp_data]
            y_vals = [x[1] for x in wcdfp_data]
            ax = axes[plot_idx]

            cmap = custom_YlGn if mode == 1 else custom_Blues
            scatter_handles = ax.scatter(
                x_vals, y_vals, c=color_values if mode in [1, 2] else "orange", s=5, cmap=cmap, vmin=0, vmax=1
            )

            # Count points above and below y = x
            above_line = sum(1 for x, y in zip(x_vals, y_vals) if y > x)
            below_line = sum(1 for x, y in zip(x_vals, y_vals) if y <= x)

            # Add labels for point counts
            ax.text(0.02, 0.98, f"{above_line}",
                    transform=ax.transAxes, fontsize=12,
                    ha='left', va='top', bbox=dict(facecolor='white', edgecolor='black', boxstyle="round,pad=0.3"))
            ax.text(0.98, 0.02, f"{below_line}",
                    transform=ax.transAxes, fontsize=12,
                    ha='right', va='bottom', bbox=dict(facecolor='white', edgecolor='black', boxstyle="round,pad=0.3"))
            
            # Add reference line
            min_val = min(min(x_vals), min(y_vals))
            max_val = max(max(x_vals), max(y_vals))
            ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="#0F4D48", zorder=10)
            ax.tick_params(axis='both', which='major', labelsize=14)

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(f"WCDFP Estimated by $\\bf{{{method_to_label[method_1]}}}$", fontsize=16)
            ax.set_ylabel(f"WCDFP Estimated by $\\bf{{{method_to_label[method_2]}}}$", fontsize=16)

            ax.grid(visible=True, which='major', linestyle='--', linewidth=0.5)
            ax.set_box_aspect(1)

            plot_idx += 1

    # Add colorbar
    if mode in [1, 2]:
        cbar = fig.colorbar(scatter_handles, ax=axes, aspect=50, shrink=0.7, pad=0.03, location="right")
        cbar.set_label(color_label, rotation=270, labelpad=10, fontsize=16)
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.yaxis.labelpad = 40

    # Save plot
    output = os.path.join(output_prefix, output_file)
    plt.savefig(output, dpi=300, format="pdf", bbox_inches='tight')
    print(f"Merged plot saved as {output_file}")


def plot_comparison_for_task_id():
    """
    Plot ExecutionTime vs WCDFP for a given TaskSetID, comparing different methods.
    """
    task_id = 35
    task_count = 50
    util_rate = 0.60
    epsilon = "0.001"

    # Define colors for each method
    method_colors = {
        "monte_carlo_multi": "tab:blue",    # Line + Points
        "monte_carlo_single": "tab:purple", # Line + Points
        "berry_essen": "tab:orange",        # Points
        "convolution": "tab:green",         # Points
        "convolution_merge": "tab:red"
    }

    # Define methods to loop through
    methods = [
        "monte_carlo_multi",
        "monte_carlo_single",
        "berry_essen",
        "convolution",
        "convolution_merge"
    ]

    # Initialize plot
    plt.figure(figsize=(10, 7))

    for method in methods:
        if method in ["monte_carlo_multi", "monte_carlo_single"]:
            input_file = f"evaluation_{method}_{task_id}.csv"
        else:
            input_file = f"evaluation_{method}.csv"
        prefix = f"{output_prefix}/{task_count}_{util_rate:.2f}_{epsilon}"
        file_path = os.path.join(prefix, input_file)

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
        print(f"Method: {method}, x: {x}, y: {y}")

        # Determine plot style based on method
        label = f"{method_to_label[method]}"  # Apply label mapping
        print(f"Label: {label}")
        if method in ["monte_carlo_multi", "monte_carlo_single"]:
            plt.plot(x, y, marker='o', linestyle='-', color=method_colors[method],
                     label=f"{label}", linewidth=1.5)
        else:
            plt.scatter(x, y, color=method_colors[method], s=60, label=f"{label}")

            # Add a dotted horizontal line for each WCDFP value
            for wcdfp_value in y:
                plt.axhline(y=wcdfp_value, xmin=0, xmax=1, color=method_colors[method],
                            linestyle='--', linewidth=1.5, alpha=0.7)

    # Set log scale for both axes
    plt.xscale("log")
    plt.yscale("log")

    # Set tick label font size for both axes
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Add labels
    plt.xlabel("Execution Time [s]", fontsize=16)
    plt.ylabel("WCDFP", fontsize=16)

    # Add grid and legend
    plt.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.minorticks_off()
    plt.legend(fontsize=16, loc='upper right')

    # Save plot
    output = os.path.join(output_prefix, f"comparison_taskset_{task_id}.pdf")
    plt.tight_layout()
    plt.savefig(output, dpi=300, format="pdf", bbox_inches='tight')
    plt.close()

    print(f"Plot saved to {output}")



def merge_two_pdfs_side_by_side():
    """
    Merge two single-page PDFs side by side into one PDF page,
    matching their heights by scaling the second PDF if needed.
    """

    pdf1_path = "src/evaluation/output_0117/merged_plot.pdf"
    pdf2_path = "src/evaluation/output_0117/execution_time_boxplot_aggregated.pdf"
    output_pdf_path = "src/evaluation/output_0117/merged_side_by_side.pdf"
    
    # Open both PDFs
    doc1 = fitz.open(pdf1_path)
    doc2 = fitz.open(pdf2_path)
    
    # For simplicity, assume each PDF has only one page
    page1 = doc1[0]
    page2 = doc2[0]
    
    # Get the page rectangles
    rect1 = page1.rect
    rect2 = page2.rect
    
    # Calculate the scale factor so that the second PDF's height
    # matches the first PDF's height
    scale2 = rect1.height / rect2.height
    
    # New width and height of the second PDF after scaling
    scaled_width2 = rect2.width * scale2
    scaled_height2 = rect2.height * scale2  # should match rect1.height
    
    # Create a new PDF to hold the merged page
    merged_doc = fitz.open()
    
    # Create a page whose width is the sum of the two widths,
    # and whose height is the same as the first PDF's height.
    merged_page = merged_doc.new_page(
        width=rect1.width + scaled_width2,
        height=rect1.height
    )
    
    # Place the first PDF page on the left (0,0) -> (rect1.width, rect1.height)
    merged_page.show_pdf_page(
        # The region (rectangle) where PDF1 will be shown
        fitz.Rect(0, 0, rect1.width, rect1.height),
        doc1,                     # Source document
        0,                        # Page number in doc1
        keep_proportion=False     # We'll use exact rect sizing
    )
    
    # Place the second PDF page on the right,
    # starting from x = rect1.width to x + scaled_width2
    merged_page.show_pdf_page(
        fitz.Rect(rect1.width + 20, 0, rect1.width + scaled_width2, scaled_height2),
        doc2,
        0,
        keep_proportion=False
    )
    
    # Save the merged PDF
    merged_doc.save(output_pdf_path)
    
    # Close all documents
    merged_doc.close()
    doc1.close()
    doc2.close()
    
    print(f"Merged PDF saved to: {output_pdf_path}")
