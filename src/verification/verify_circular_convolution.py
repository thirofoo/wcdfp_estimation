import os
import numpy as np
import matplotlib.pyplot as plt
from common.parameters import MINIMUM_TIME_UNIT
from common.taskset import TaskSet
from methods.circular_convolution.estimation import (
    calculate_wcdfp_by_sequential_conv,
    calculate_wcdfp_by_aggregate_conv_orig,
    calculate_wcdfp_by_aggregate_conv_imp
)

def calculate_cdf(response_time):
    # Compute the cumulative distribution function for the response time
    return np.cumsum(response_time) / np.sum(response_time)

def run_convolution_methods(taskset, target_job):
    # List of methods to calculate response time with their labels and options
    methods = [
        {"label": "Sequential", "func": calculate_wcdfp_by_sequential_conv, "kwargs": {"log_flag": True}},
        {"label": "Aggregate (Orig)", "func": calculate_wcdfp_by_aggregate_conv_orig, "kwargs": {}},
        {"label": "Aggregate (Imp)", "func": calculate_wcdfp_by_aggregate_conv_imp, "kwargs": {}}
    ]

    results = {}
    for method in methods:
        label = method["label"]
        print(f"Calculating response time using {label} convolution...")
        response_time, wcdfp = method["func"](taskset, target_job, **method["kwargs"])
        results[label] = {
            "cdf": calculate_cdf(response_time),
            "wcdfp": wcdfp
        }
        print(f"WCDFP ({label}): {wcdfp}\n")
    return results

def plot_response_time_cdfs(cdfs, deadline, output_dir):
    # Find the minimum length among all CDF arrays
    min_length = min(len(cdf) for cdf in cdfs.values())
    time_indices = np.arange(min_length) * MINIMUM_TIME_UNIT

    # Log lengths for debugging
    lengths = {label: len(cdf) for label, cdf in cdfs.items()}
    print(f"Time indices length: {len(time_indices)}, CDF lengths: {lengths}")

    plt.figure(figsize=(10, 6))
    styles = {
        "Sequential": {"color": "blue",   "linestyle": "--", "label": "Sequential CDF"},
        "Aggregate (Orig)": {"color": "orange", "linestyle": "-",  "label": "Aggregate (Orig) CDF"},
        "Aggregate (Imp)": {"color": "green",  "linestyle": "-.", "label": "Aggregate (Imp) CDF"}
    }
    for label, style in styles.items():
        plt.plot(
            time_indices,
            cdfs[label][:min_length],
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            drawstyle="steps-mid"
        )
    plt.axvline(x=deadline, color="red", linestyle="--", label="Deadline")

    plt.xlabel("Response Time")
    plt.ylabel("Cumulative Distribution Function")
    plt.title("Response Time CDFs (Convolution vs Aggregate (Orig) vs Aggregate (Imp))")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    plot_path = os.path.join(output_dir, "response_time_cdfs.png")
    plt.savefig(plot_path)
    print(f"Response Time CDFs plot saved to {plot_path}")
    plt.close()

def verify_convolution():
    # Create a task set and output directory
    seed = 13
    task_num = 10
    utilization_rate = 0.70
    taskset = TaskSet(task_num, utilization_rate, seed=seed)

    output_dir = "src/verification/output"
    os.makedirs(output_dir, exist_ok=True)

    target_job = taskset.target_job
    print(f"Lowest priority absolute_deadline: {target_job.absolute_deadline}")
    print(f"Lowest priority execution_time: {target_job.task.get_execution_time()}\n")

    # Run all convolution methods and collect results
    results = run_convolution_methods(taskset, target_job)

    # Collect CDFs for plotting
    cdfs = {label: data["cdf"] for label, data in results.items()}
    plot_response_time_cdfs(cdfs, target_job.absolute_deadline, output_dir)
