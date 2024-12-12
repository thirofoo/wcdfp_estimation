import os
import matplotlib.pyplot as plt
import numpy as np
from common.taskset import TaskSet
from common.utils import calculate_expected_execution_time
from contextlib import redirect_stdout
from tqdm import tqdm

def verify_taskset():
    """
    Verify Taskset generation and visualize execution time distribution.
    """
    # ==================== Generate Taskset using TaskSet class ==================== #
    seed = 3
    task_num = 100
    utilization_rate = 0.65

    # Set output directory
    output_dir = "src/verification/output"
    os.makedirs(output_dir, exist_ok=True)

    # Output timeline size
    log_path = os.path.join(output_dir, "execution_log.txt")
    with open(log_path, "w") as log_file:
        with redirect_stdout(log_file):
            # TaskSet constructor's output will be redirected here
            taskset = TaskSet(task_num=task_num, utilization_rate=utilization_rate, seed=seed)
        log_file.write(f"Timeline size: {len(taskset.timeline)}\n\n")
        
        # Output taskset information
        log_file.write("Task Set:\n")
        for idx, task in enumerate(taskset.tasks):
            log_file.write(f"WCET: {task.wcet}, Relative Deadline: {task.relative_deadline}, Min Inter-Arrival: {task.minimum_inter_arrival_time}\n")

        # Output the first 10 entries of the timeline
        log_file.write("\nTimeline:\n")
        print_time = 0
        for t, jobs_at_t in enumerate(taskset.timeline):
            if print_time > 10:
                break
            if jobs_at_t:
                log_file.write(f"Time {t}: {[f'Job Execution Time: {job.task.get_execution_time()}' for job in jobs_at_t]}\n")
                print_time += 1

    print(f"Execution log saved to {log_path}")

    # Take samples and display the distribution
    sample_count = 10000
    task_idx = -1  # Index of the target task
    samples = [taskset.tasks[task_idx].get_execution_time() for _ in range(sample_count)]

    # Calculate the mean of the samples
    sample_mean = np.mean(samples)

    # Display and save the histogram of execution times
    hist_path = os.path.join(output_dir, "execution_time_distribution.png")
    plt.hist(
        samples,
        bins=100,
        edgecolor='black',
        alpha=0.7,
        density=True
    )
    plt.xlabel('Execution Time')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Execution Times (mean = {sample_mean:.2f})')
    plt.savefig(hist_path)
    print(f"Execution time distribution histogram saved to {hist_path}")
    plt.close()

    # Calculate the true mean using the expected execution time
    trunc_lower = 0
    trunc_upper = taskset.tasks[task_idx].wcet  # Set WCET as the upper limit of the range
    true_mean = calculate_expected_execution_time(
        wcet=taskset.tasks[task_idx].wcet,
        trunc_lower=trunc_lower,
        trunc_upper=trunc_upper
    )

    # Output mean and utilization rate to the log
    with open(log_path, "a") as log_file:
        log_file.write(f"\nSample Mean: {sample_mean}\n")
        log_file.write(f"True Mean (based on expected execution time): {true_mean}\n")
        log_file.write(f"Average utilization rate (based on sample): {sample_mean / taskset.tasks[task_idx].minimum_inter_arrival_time}\n")
        log_file.write(f"Average utilization rate (based on true mean): {true_mean / taskset.tasks[task_idx].minimum_inter_arrival_time}\n")


def verify_taskset_consistency():
    """
    Generate task sets with different configurations and verify consistency of relative deadlines
    and periods for all tasks.
    Log any discrepancies.
    """
    output_dir = "src/verification/output"
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, "taskset_consistency_log.txt")
    with open(log_path, "w") as log_file:
        
        # Configurations
        utilization_rates = [0.60, 0.65, 0.70]
        task_counts = range(10, 101, 10)
        seed_range = range(1, 51)

        total_iterations = len(utilization_rates) * len(task_counts) * len(seed_range)

        # Loop through configurations with progress tracking
        with tqdm(total=total_iterations, desc="Verifying Tasksets") as pbar:
            for utilization_rate in utilization_rates:
                for task_count in task_counts:
                    for seed in seed_range:

                        # Generate taskset
                        taskset = TaskSet(task_num=task_count, utilization_rate=utilization_rate, seed=seed)

                        # Check consistency for each task
                        for idx, task in enumerate(taskset.tasks):
                            if task.relative_deadline != task.minimum_inter_arrival_time:
                                log_file.write(
                                    f"Discrepancy Found: Utilization={utilization_rate}, TaskCount={task_count}, Seed={seed}, "
                                    f"TaskIndex={idx}, RelativeDeadline={task.relative_deadline}, Period={task.minimum_inter_arrival_time}\n"
                                )
                        pbar.update(1)

    print(f"Taskset consistency verification log saved to {log_path}")
