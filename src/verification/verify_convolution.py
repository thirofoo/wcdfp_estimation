import os
import numpy as np
import matplotlib.pyplot as plt
from common.parameters import MINIMUM_TIME_UNIT
from common.taskset import TaskSet
from methods.circular_convolution.estimation import calculate_response_time_by_conv, compute_response_time_with_doubling

def plot_response_time_cdfs(cdf_conv, cdf_doubling, deadline, output_dir):
    """
    Plot the response time CDFs and save the plot to the output directory.

    :param cdf_conv: CDF from convolution method
    :param cdf_doubling: CDF from doubling method
    :param deadline: Absolute deadline of the target job
    :param output_dir: Directory to save the plot
    """
    time_indices = np.arange(len(cdf_conv)) * MINIMUM_TIME_UNIT

    plt.figure(figsize=(10, 6))
    plt.plot(time_indices, cdf_conv, label="Convolution CDF", color="blue", linestyle="--", drawstyle="steps-mid")
    plt.plot(time_indices, cdf_doubling, label="Doubling CDF", color="orange", linestyle="-", drawstyle="steps-mid")
    plt.axvline(x=deadline, color="red", linestyle="--", label="Deadline")

    plt.xlabel("Response Time")
    plt.ylabel("Cumulative Distribution Function")
    plt.title("Response Time CDFs (Convolution vs Doubling)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    plot_path = os.path.join(output_dir, "response_time_cdfs.png")
    plt.savefig(plot_path)
    print(f"Response Time CDFs plot saved to {plot_path}")
    plt.close()

def verify_convolution():
    """
    Verify response time calculation using convolution for the lowest priority task in a TaskSet.
    """
    # Create TaskSet
    seed = 3
    task_num = 100
    utilization_rate = 0.60
    taskset = TaskSet(task_num, utilization_rate, seed=seed)

    # Prepare output directory
    output_dir = "src/verification/output"
    os.makedirs(output_dir, exist_ok=True)

    # Log details about the lowest priority task's job
    target_job = taskset.target_job
    print(f"lowest priority absolute_deadline : {target_job.absolute_deadline}")
    print(f"lowest priority execution_time : {target_job.task.get_execution_time()}\n")

    # Calculate response time using convolution
    print("Calculating response time using basic convolution...")
    response_time_conv, wcdfp_conv = calculate_response_time_by_conv(taskset, target_job, log_flag=True)
    cdf_conv = np.cumsum(response_time_conv) / np.sum(response_time_conv)
    print(f"WCDFP (Convolution): {wcdfp_conv}\n")

    # Calculate response time using doubling
    print("Calculating response time using doubling convolution...")
    response_time_doubling, wcdfp_doubling = compute_response_time_with_doubling(taskset, target_job)
    cdf_doubling = np.cumsum(response_time_doubling) / np.sum(response_time_doubling)
    print(f"WCDFP (Doubling): {wcdfp_doubling}")

    # Plot and save CDFs
    plot_response_time_cdfs(cdf_conv, cdf_doubling, target_job.absolute_deadline, output_dir)

if __name__ == "__main__":
    verify_convolution()
