import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from methods.monte_carlo.estimation import calculate_response_time_distribution, required_sample_size_binomial
from methods.berry_essen.estimation import calculate_response_time_by_berry_essen
from methods.circular_convolution.estimation import calculate_response_time_with_doubling, calculate_response_time_by_conv
from common.taskset import TaskSet
from common.parameters import MINIMUM_TIME_UNIT, BERRY_ESSEN_COEFFICIENT as A

# Parameters
default_total_taskset = 50
default_log_flag = False
default_plot_flag = False
default_task_num = 100
default_utilization_rate = 0.65
default_thread_num = 16
default_false_probability = 0.000001
default_error_margin = 0.01
default_float128_flag = True

output_prefix = "src/evaluation/output"

def evaluate_monte_carlo(
    task_num=default_task_num,
    utilization_rate=default_utilization_rate,
    total_taskset=default_total_taskset,
    thread_num=default_thread_num,
    false_probability=default_false_probability,
    error_margin=default_error_margin,
    log_flag=default_log_flag,
    plot_flag=default_plot_flag,
):
    # Calculate required sample size
    samples = required_sample_size_binomial(error_margin, false_probability)

    # Output directory for results
    output_dir = f"{output_prefix}/{task_num}_{utilization_rate}_{MINIMUM_TIME_UNIT}"
    os.makedirs(output_dir, exist_ok=True)

    # CSV file path
    results_path = os.path.join(output_dir, "evaluation_monte_carlo.csv")

    # Load existing results if the CSV file exists
    if os.path.exists(results_path):
        existing_results = pd.read_csv(results_path)
    else:
        existing_results = pd.DataFrame(columns=["TaskSetID", "TaskNum", "UtilizationRate", "WCDFP", "ExecutionTime", "Samples", "DeadlineMissCount"])

    # Identify evaluated TaskSetIDs
    evaluated_taskset_ids = set(existing_results["TaskSetID"].values)

    with tqdm(total=total_taskset, desc="Evaluating TaskSets", unit="set") as progress_bar:
        for i in range(total_taskset):
            taskset_id = i + 1

            # Skip already evaluated TaskSets
            if taskset_id in evaluated_taskset_ids:
                progress_bar.update(1)
                continue
            
            progress_bar.set_description(f"Evaluating TaskSet {taskset_id}/{total_taskset}")

            # Generate TaskSet
            taskset = TaskSet(task_num=task_num, utilization_rate=utilization_rate, seed=taskset_id)

            # Calculate Monte Carlo response time distribution
            start_time = time.time()
            # Update the seed to ensure non-overlapping seeds for threads in parallel processing
            seed = i * thread_num
            response_times_mc, wcdfp_mc = calculate_response_time_distribution(
                taskset=taskset,
                target_job=taskset.target_job,
                false_probability=false_probability,
                thread_num=thread_num,
                log_flag=log_flag,
                plot_flag=plot_flag,
                samples=samples,
                seed=seed,
            )
            elapsed_time = time.time() - start_time

            # Calculate deadline misses
            absolute_deadline = taskset.timeline[0][-1].absolute_deadline
            response_times_mc = np.array(response_times_mc)
            deadline_miss_count = np.sum(response_times_mc > absolute_deadline)

            # Save results for the current TaskSet
            result = {
                "TaskSetID": taskset_id,
                "TaskNum": task_num,
                "UtilizationRate": utilization_rate,
                "WCDFP": wcdfp_mc,
                "ExecutionTime": elapsed_time,
                "Samples": samples,
                "DeadlineMissCount": deadline_miss_count
            }

            # Append the result directly to the CSV file
            pd.DataFrame([result]).to_csv(results_path, mode='a', index=False, header=not os.path.exists(results_path))

            # Plot and save figure
            # sorted_times_mc = np.sort(response_times_mc)
            # monte_carlo_cdf_values = np.arange(1, len(sorted_times_mc) + 1) / len(sorted_times_mc)
            # plt.figure(figsize=(10, 6))
            # plt.step(sorted_times_mc, monte_carlo_cdf_values, label="Monte Carlo CDF", where="post")
            # deadline = taskset.timeline[0][-1].absolute_deadline
            # plt.axvline(x=deadline, color="red", linestyle="--", label="Deadline")
            # plt.xlabel("Execution Time")
            # plt.ylabel("Cumulative Distribution Function")
            # plt.title(f"Monte Carlo CDF for TaskSet {i + 1}")
            # plt.legend()
            # plt.grid(True)
            # plot_path = os.path.join(output_dir, f"monte_carlo_cdf_{i + 1}.png")
            # plt.savefig(plot_path)
            # plt.close()

            progress_bar.update(1)

    print(f"Results saved incrementally to {results_path}")
    print("\nMonte Carlo evaluation completed.")


def evaluate_berry_essen(
    task_num=default_task_num,
    utilization_rate=default_utilization_rate,
    total_taskset=default_total_taskset,
    log_flag=default_log_flag,
):
    # Output directory for results
    output_dir = f"{output_prefix}/{task_num}_{utilization_rate}_{MINIMUM_TIME_UNIT}"
    os.makedirs(output_dir, exist_ok=True)

    # CSV file path
    results_path = os.path.join(output_dir, "evaluation_berry_essen.csv")

    # Load existing results if the CSV file exists
    if os.path.exists(results_path):
        existing_results = pd.read_csv(results_path)
    else:
        existing_results = pd.DataFrame(columns=["TaskSetID", "TaskNum", "UtilizationRate", "WCDFP", "ExecutionTime"])

    # Identify evaluated TaskSetIDs
    evaluated_taskset_ids = set(existing_results["TaskSetID"].values)

    with tqdm(total=total_taskset, desc="Evaluating TaskSets", unit="set") as progress_bar:
        for i in range(total_taskset):
            taskset_id = i + 1

            # Skip already evaluated TaskSets
            if taskset_id in evaluated_taskset_ids:
                progress_bar.update(1)
                continue

            progress_bar.set_description(f"Evaluating TaskSet {taskset_id}/{total_taskset}")

            # Generate TaskSet
            taskset = TaskSet(task_num=task_num, utilization_rate=utilization_rate, seed=taskset_id)

            # Calculate Berry-Essen WCDFP
            start_time = time.time()
            wcdfp_be, (x_values, berry_essen_cdf_values) = calculate_response_time_by_berry_essen(
                taskset=taskset,
                target_job=taskset.target_job,
                A=A,
                upper=True,
                log_flag=log_flag,
                seed=i
            )
            elapsed_time = time.time() - start_time

            # Save results for the current TaskSet
            result = {
                "TaskSetID": taskset_id,
                "TaskNum": task_num,
                "UtilizationRate": utilization_rate,
                "WCDFP": wcdfp_be,
                "ExecutionTime": elapsed_time,
            }

            # Append the result directly to the CSV file
            pd.DataFrame([result]).to_csv(results_path, mode='a', index=False, header=not os.path.exists(results_path))

            # Plot and save figure
            # plt.figure(figsize=(10, 6))
            # plt.plot(x_values, berry_essen_cdf_values, label="Berry-Essen CDF")
            # absolute_deadline = taskset.timeline[0][-1].absolute_deadline
            # plt.axvline(x=absolute_deadline, color="red", linestyle="--", label="Deadline")
            # plt.xlabel("Response Time")
            # plt.ylabel("Cumulative Distribution Function")
            # plt.title(f"Berry-Essen CDF for TaskSet {taskset_id}")
            # plt.legend()
            # plt.grid(True)
            # plot_path = os.path.join(output_dir, f"berry_essen_cdf_{taskset_id}.png")
            # plt.savefig(plot_path)
            # plt.close()

            progress_bar.update(1)

    print(f"Results saved incrementally to {results_path}")
    print("\nBerry-Essen evaluation completed.")


def evaluate_convolution_doubling(
    task_num=default_task_num,
    utilization_rate=default_utilization_rate,
    total_taskset=default_total_taskset,
    thread_num=default_thread_num,
    log_flag=default_log_flag,
    float128_flag=default_float128_flag,
):
    # Output directory for results
    output_dir = f"{output_prefix}/{task_num}_{utilization_rate}_{MINIMUM_TIME_UNIT}"
    os.makedirs(output_dir, exist_ok=True)

    # CSV file path
    results_path = os.path.join(output_dir, "evaluation_convolution_doubling.csv")

    # Load existing results if the CSV file exists
    if os.path.exists(results_path):
        existing_results = pd.read_csv(results_path)
    else:
        existing_results = pd.DataFrame(columns=["TaskSetID", "TaskNum", "UtilizationRate", "WCDFP", "ExecutionTime", "DeadlineMissCount"])

    # Identify evaluated TaskSetIDs
    evaluated_taskset_ids = set(existing_results["TaskSetID"].values)

    with tqdm(total=total_taskset, desc="Evaluating TaskSets", unit="set") as progress_bar:
        for i in range(total_taskset):
            taskset_id = i + 1

            # Skip already evaluated TaskSets
            if taskset_id in evaluated_taskset_ids:
                progress_bar.update(1)
                continue

            # Update the seed to ensure non-overlapping seeds for threads in parallel processing
            seed = i * thread_num
            progress_bar.set_description(f"Evaluating TaskSet {taskset_id}/{total_taskset}")

            # Generate TaskSet
            taskset = TaskSet(task_num=task_num, utilization_rate=utilization_rate, seed=taskset_id)

            # Calculate response time using convolution doubling
            start_time = time.time()
            response_time, wcdfp_conv = calculate_response_time_with_doubling(
                taskset=taskset,
                target_job=taskset.target_job,
                log_flag=log_flag,
                float128_flag=float128_flag,
            )
            elapsed_time = time.time() - start_time

            # Save results for the current TaskSet
            result = {
                "TaskSetID": taskset_id,
                "TaskNum": task_num,
                "UtilizationRate": utilization_rate,
                "WCDFP": wcdfp_conv,
                "ExecutionTime": elapsed_time,
            }

            # Append the result directly to the CSV file
            pd.DataFrame([result]).to_csv(results_path, mode='a', index=False, header=not os.path.exists(results_path))

            # Calculate and Plot CDF from PDF
            # absolute_deadline = taskset.timeline[0][-1].absolute_deadline
            # cdf = np.cumsum(response_time)  # Cumulative sum to compute the CDF
            # cdf /= cdf[-1]  # Normalize to ensure the CDF ranges from 0 to 1
            # time_indices = np.arange(len(response_time)) * MINIMUM_TIME_UNIT
            # plt.figure(figsize=(10, 6))
            # plt.plot(time_indices, cdf, drawstyle="steps-mid", label="Convolution Doubling CDF")
            # plt.axvline(x=absolute_deadline, color="red", linestyle="--", label="Deadline")
            # plt.xlabel("Response Time")
            # plt.ylabel("Cumulative Probability")
            # plt.title(f"Convolution Doubling CDF for TaskSet {taskset_id}")
            # plt.legend()
            # plt.grid(True)
            # plot_path = os.path.join(output_dir, f"convolution_doubling_cdf_{taskset_id}.png")
            # plt.savefig(plot_path)
            # plt.close()

            progress_bar.update(1)

    print(f"Results saved incrementally to {results_path}")
    print("\nConvolution Doubling evaluation completed.")


def evaluate_convolution(
    task_num=default_task_num,
    utilization_rate=default_utilization_rate,
    total_taskset=default_total_taskset,
    thread_num=default_thread_num,
    log_flag=default_log_flag,
    float128_flag=default_float128_flag,
):
    # Output directory for results
    output_dir = f"{output_prefix}/{task_num}_{utilization_rate}_{MINIMUM_TIME_UNIT}"
    os.makedirs(output_dir, exist_ok=True)

    # CSV file path
    results_path = os.path.join(output_dir, "evaluation_convolution.csv")

    # Load existing results if the CSV file exists
    if os.path.exists(results_path):
        existing_results = pd.read_csv(results_path)
    else:
        existing_results = pd.DataFrame(columns=["TaskSetID", "TaskNum", "UtilizationRate", "WCDFP", "ExecutionTime", "DeadlineMissCount"])

    # Identify evaluated TaskSetIDs
    evaluated_taskset_ids = set(existing_results["TaskSetID"].values)

    with tqdm(total=total_taskset, desc="Evaluating TaskSets", unit="set") as progress_bar:
        for i in range(total_taskset):
            taskset_id = i + 1

            # Skip already evaluated TaskSets
            if taskset_id in evaluated_taskset_ids:
                progress_bar.update(1)
                continue

            # Update the seed to ensure non-overlapping seeds for threads in parallel processing
            seed = i * thread_num
            progress_bar.set_description(f"Evaluating TaskSet {taskset_id}/{total_taskset}")

            # Generate TaskSet
            taskset = TaskSet(task_num=task_num, utilization_rate=utilization_rate, seed=taskset_id)

            # Calculate response time using convolution
            start_time = time.time()
            response_time, wcdfp_conv = calculate_response_time_by_conv(
                taskset=taskset,
                target_job=taskset.target_job,
                log_flag=log_flag,
                float128_flag=float128_flag,
            )
            elapsed_time = time.time() - start_time

            # Save results for the current TaskSet
            result = {
                "TaskSetID": taskset_id,
                "TaskNum": task_num,
                "UtilizationRate": utilization_rate,
                "WCDFP": wcdfp_conv,
                "ExecutionTime": elapsed_time,
            }

            # Append the result directly to the CSV file
            pd.DataFrame([result]).to_csv(results_path, mode='a', index=False, header=not os.path.exists(results_path))

            # Calculate and Plot CDF from PDF
            # absolute_deadline = taskset.timeline[0][-1].absolute_deadline
            # cdf = np.cumsum(response_time)  # Cumulative sum to compute the CDF
            # cdf /= cdf[-1]  # Normalize to ensure the CDF ranges from 0 to 1
            # time_indices = np.arange(len(response_time)) * MINIMUM_TIME_UNIT
            # plt.figure(figsize=(10, 6))
            # plt.plot(time_indices, cdf, drawstyle="steps-mid", label="Convolution CDF")
            # plt.axvline(x=absolute_deadline, color="red", linestyle="--", label="Deadline")
            # plt.xlabel("Response Time")
            # plt.ylabel("Cumulative Probability")
            # plt.title(f"Convolution CDF for TaskSet {taskset_id}")
            # plt.legend()
            # plt.grid(True)
            # plot_path = os.path.join(output_dir, f"convolution_cdf_{taskset_id}.png")
            # plt.savefig(plot_path)
            # plt.close()

            progress_bar.update(1)

    print(f"Results saved incrementally to {results_path}")
    print("\nConvolution evaluation completed.")


def evaluate_all_methods():
    """
    Evaluate all methods for all combinations of task_num and utilization_rate.
    """
    all_task_num = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    all_utilization_rate = [0.60, 0.65, 0.70]

    for t_num in all_task_num:  # Iterate over task numbers
        for u_rate in all_utilization_rate:  # Iterate over utilization rates
            print(f"Evaluating for TaskNum: {t_num}, UtilizationRate: {u_rate}")

            # Pass the current parameters to each evaluation function
            evaluate_monte_carlo(task_num=t_num, utilization_rate=u_rate)
            evaluate_berry_essen(task_num=t_num, utilization_rate=u_rate)
            evaluate_convolution_doubling(task_num=t_num, utilization_rate=u_rate)
            evaluate_convolution(task_num=t_num, utilization_rate=u_rate)


def plot_normalized_response_times(
    task_num=10,
    utilization_rate=0.60,
    false_probability=1e-6,
    error_margin=0.01,
    float128_flag=default_float128_flag,
):
    """
    Evaluate a single task set with all methods, normalize the response times,
    and plot them together on one graph.
    """
    # Define task set parameters
    taskset_seed = 47
    taskset = TaskSet(task_num=task_num, utilization_rate=utilization_rate, seed=taskset_seed)

    # Evaluate each method and collect normalized response time distributions
    methods_data = []

    # Monte Carlo
    start_time = time.time()
    samples = required_sample_size_binomial(error_margin, false_probability)

    response_times_mc, wcdfp_mc = calculate_response_time_distribution(
        taskset=taskset,
        target_job=taskset.target_job,
        false_probability=false_probability,
        thread_num=16,
        log_flag=False,
        plot_flag=False,
        samples=samples,
        seed=(taskset_seed - 1) * 16,
    )
    response_times_mc = np.sort(response_times_mc)
    cdf_mc = np.arange(1, len(response_times_mc) + 1) / len(response_times_mc)
    methods_data.append(("Monte Carlo", response_times_mc, cdf_mc))
    print(f"Monte Carlo evaluation completed in {time.time() - start_time:.2f} seconds.")

    # Convolution
    start_time = time.time()
    response_time_conv, wcdfp_conv = calculate_response_time_by_conv(
        taskset,
        taskset.target_job,
        float128_flag=float128_flag
    )
    response_time_conv = np.concatenate([response_time_conv, [wcdfp_conv]])
    if isinstance(response_time_conv, list):
        response_time_conv = np.concatenate(response_time_conv)
    response_time_conv = np.asarray(response_time_conv)  # Ensure numpy array for compatibility
    response_time_conv_cdf = np.cumsum(response_time_conv) / np.sum(response_time_conv)
    conv_indices = np.arange(len(response_time_conv)) * MINIMUM_TIME_UNIT
    methods_data.append(("Convolution", conv_indices, response_time_conv_cdf))
    print(f"Convolution evaluation completed in {time.time() - start_time:.2f} seconds.")

    # Doubling Convolution
    start_time = time.time()
    response_time_doubling, wcdfp_doubling = calculate_response_time_with_doubling(
        taskset,
        taskset.target_job,
        float128_flag=float128_flag
    )
    response_time_doubling = np.concatenate([response_time_doubling, [wcdfp_doubling]])
    if isinstance(response_time_doubling, list):
        response_time_doubling = np.concatenate(response_time_doubling)
    response_time_doubling = np.asarray(response_time_doubling)  # Ensure numpy array for compatibility
    response_time_doubling_cdf = np.cumsum(response_time_doubling) / np.sum(response_time_doubling)
    doubling_indices = np.arange(len(response_time_doubling)) * MINIMUM_TIME_UNIT
    methods_data.append(("Doubling", doubling_indices, response_time_doubling_cdf))
    print(f"Doubling Convolution evaluation completed in {time.time() - start_time:.2f} seconds.")

    # Berry-Essen
    start_time = time.time()
    wcdfp_be, (x_values, berry_essen_cdf_values) = calculate_response_time_by_berry_essen(
        taskset=taskset,
        target_job=taskset.target_job,
        A=A,
        upper=True,
        log_flag=False,
        seed=(taskset_seed - 1)
    )
    methods_data.append(("Berry-Essen", x_values, berry_essen_cdf_values))
    print(f"Berry-Essen evaluation completed in {time.time() - start_time:.2f} seconds.")

    print(f"WCDFP (Monte Carlo): {wcdfp_mc}")
    print(f"WCDFP (Convolution): {wcdfp_conv}")
    print(f"WCDFP (Doubling): {wcdfp_doubling}")
    print(f"WCDFP (Berry-Essen): {wcdfp_be}")

    # Normalize and plot
    plt.figure(figsize=(12, 8))
    for label, times, cdf in methods_data:
        plt.step(times, cdf, label=label, where="post")

    # Add deadline line
    deadline = taskset.target_job.absolute_deadline
    plt.axvline(x=deadline, color="red", linestyle="--", label="Deadline")

    # Configure plot
    plt.xlabel("Normalized Response Time")
    plt.ylabel("Cumulative Distribution Function")
    plt.title("Normalized Response Time CDFs Across Methods")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Save and show the plot
    output_path = os.path.join(output_prefix, "response_time_cdfs_normalized.png")
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")
