import os
import numpy as np
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
from tqdm import tqdm

from common.taskset import TaskSet
from common.utils import calculate_expected_execution_time

OUTPUT_DIR = "src/verification/output"


def setup_output_dir(directory):
    """Create the output directory if it does not exist."""
    os.makedirs(directory, exist_ok=True)


def save_execution_log(taskset, log_path):
    """Write taskset details and timeline entries to the log file."""
    with open(log_path, "a") as log_file:
        log_file.write(f"Timeline size: {len(taskset.timeline)}\n\n")
        log_file.write("Task Set:\n")
        for idx, task in enumerate(taskset.tasks):
            log_file.write(
                f"WCET: {task.wcet}, Relative Deadline: {task.relative_deadline}, Min Inter-Arrival: {task.minimum_inter_arrival_time}\n"
            )
        log_file.write("\nTimeline:\n")
        printed_times = 0
        for t, jobs_at_t in enumerate(taskset.timeline):
            if printed_times > 10:
                break
            if jobs_at_t:
                job_times = [f"Job Execution Time: {job.task.get_execution_time()}" for job in jobs_at_t]
                log_file.write(f"Time {t}: {job_times}\n")
                printed_times += 1


def plot_and_save_histogram(samples, hist_path, sample_mean):
    """Plot the histogram of execution times and save the plot."""
    plt.hist(samples, bins=100, edgecolor="black", alpha=0.7, density=True)
    plt.xlabel("Execution Time")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Execution Times (mean = {sample_mean:.2f})")
    plt.savefig(hist_path)
    plt.close()


def verify_taskset():
    """
    Verify Taskset generation and visualize execution time distribution.
    """
    seed = 3
    task_num = 100
    utilization_rate = 0.65

    setup_output_dir(OUTPUT_DIR)
    log_path = os.path.join(OUTPUT_DIR, "execution_log.txt")

    # Capture TaskSet constructor output by redirecting stdout to the log file
    with open(log_path, "w") as log_file:
        with redirect_stdout(log_file):
            taskset = TaskSet(task_num=task_num, utilization_rate=utilization_rate, seed=seed)

    # Append taskset details and timeline information
    save_execution_log(taskset, log_path)
    print(f"Execution log saved to {log_path}")

    # Generate samples and plot execution time distribution
    sample_count = 10000
    task_idx = -1  # Target task index
    samples = [taskset.tasks[task_idx].get_execution_time() for _ in range(sample_count)]
    sample_mean = np.mean(samples)
    hist_path = os.path.join(OUTPUT_DIR, "execution_time_distribution.png")
    plot_and_save_histogram(samples, hist_path, sample_mean)
    print(f"Execution time distribution histogram saved to {hist_path}")

    # Calculate true mean execution time based on expected execution time
    trunc_lower = 0
    trunc_upper = taskset.tasks[task_idx].wcet
    true_mean = calculate_expected_execution_time(
        wcet=taskset.tasks[task_idx].wcet,
        trunc_lower=trunc_lower,
        trunc_upper=trunc_upper
    )

    # Append sample mean, true mean, and utilization rate information to the log file
    with open(log_path, "a") as log_file:
        log_file.write(f"\nSample Mean: {sample_mean}\n")
        log_file.write(f"True Mean (based on expected execution time): {true_mean}\n")
        log_file.write(
            f"Average utilization rate (based on sample): {sample_mean / taskset.tasks[task_idx].minimum_inter_arrival_time}\n"
        )
        log_file.write(
            f"Average utilization rate (based on true mean): {true_mean / taskset.tasks[task_idx].minimum_inter_arrival_time}\n"
        )


def verify_taskset_consistency():
    """
    Generate task sets with various configurations and verify the consistency of relative deadlines and periods.
    Log any discrepancies.
    """
    setup_output_dir(OUTPUT_DIR)
    log_path = os.path.join(OUTPUT_DIR, "taskset_consistency_log.txt")

    # Define parameter configurations
    utilization_rates = [0.60, 0.65, 0.70]
    task_counts = range(10, 101, 10)
    seed_range = range(1, 51)
    total_iterations = len(utilization_rates) * len(task_counts) * len(seed_range)

    with open(log_path, "w") as log_file:
        with tqdm(total=total_iterations, desc="Verifying Tasksets") as pbar:
            for utilization_rate in utilization_rates:
                for task_count in task_counts:
                    for seed in seed_range:
                        taskset = TaskSet(task_num=task_count, utilization_rate=utilization_rate, seed=seed)
                        for idx, task in enumerate(taskset.tasks):
                            if task.relative_deadline != task.minimum_inter_arrival_time:
                                log_file.write(
                                    f"Discrepancy Found: Utilization={utilization_rate}, TaskCount={task_count}, Seed={seed}, "
                                    f"TaskIndex={idx}, RelativeDeadline={task.relative_deadline}, Period={task.minimum_inter_arrival_time}\n"
                                )
                        pbar.update(1)
    print(f"Taskset consistency verification log saved to {log_path}")
