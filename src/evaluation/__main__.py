import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from methods.monte_carlo.estimation import (
    calculate_response_time_distribution,
    required_sample_size_binomial,
)
from methods.berry_essen.estimation import calculate_response_time_by_berry_essen
from methods.circular_convolution.estimation import (
    calculate_response_time_by_conv,
    calculate_response_time_with_doubling,
    calculate_response_time_with_merge,
)
from common.taskset import TaskSet
from common.parameters import MINIMUM_TIME_UNIT, BERRY_ESSEN_COEFFICIENT as A

# Parameters
default_total_taskset = 50
default_log_flag = False
default_plot_flag = False
default_task_num = 10
default_utilization_rate = 0.65
default_thread_num = 16
default_false_probability = 0.000001
default_error_margin = 0.01
default_float128_flag = False
output_prefix = "src/evaluation/output"


# Helper functions
def get_output_dir(task_num, utilization_rate):
    dir_path = os.path.join(output_prefix, f"{task_num}_{utilization_rate:.2f}_{MINIMUM_TIME_UNIT}")
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def load_existing_results(path, columns):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=columns)


def append_result(results_path, result):
    pd.DataFrame([result]).to_csv(results_path, mode="a", index=False, header=not os.path.exists(results_path))


def count_deadline_misses(response_times, absolute_deadline):
    response_times = np.array(response_times)
    return np.sum(response_times > absolute_deadline)


# Evaluation functions
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
    samples = 100000  # Or use: required_sample_size_binomial(error_margin, false_probability)
    output_dir = get_output_dir(task_num, utilization_rate)
    results_path = os.path.join(output_dir, "evaluation_monte_carlo.csv")
    existing_results = load_existing_results(results_path, 
                      ["TaskSetID", "TaskNum", "UtilizationRate", "WCDFP", "ExecutionTime", "Samples", "DeadlineMissCount"])
    evaluated_ids = set(existing_results["TaskSetID"].values)

    with tqdm(total=total_taskset, desc="Evaluating TaskSets", unit="set") as progress_bar:
        for i in range(total_taskset):
            taskset_id = i + 1
            if taskset_id in evaluated_ids:
                progress_bar.update(1)
                continue

            progress_bar.set_description(f"Evaluating TaskSet {taskset_id}/{total_taskset}")
            taskset = TaskSet(task_num=task_num, utilization_rate=utilization_rate, seed=taskset_id)
            seed = i * thread_num
            start_time = time.time()
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
            absolute_deadline = taskset.timeline[0][-1].absolute_deadline
            miss_count = count_deadline_misses(response_times_mc, absolute_deadline)

            result = {
                "TaskSetID": taskset_id,
                "TaskNum": task_num,
                "UtilizationRate": utilization_rate,
                "WCDFP": wcdfp_mc,
                "ExecutionTime": elapsed_time,
                "Samples": samples,
                "DeadlineMissCount": miss_count,
            }
            append_result(results_path, result)
            progress_bar.update(1)

    print(f"Monte Carlo results saved to {results_path}")


def evaluate_monte_carlo_adjust_sample():
    task_num = 50
    utilization_rate = 0.60
    taskset_id = 1
    thread_num = default_thread_num
    log_flag = default_log_flag
    plot_flag = default_plot_flag
    false_probability = default_false_probability

    sample_sizes = [10 ** i for i in range(9)]
    output_dir = get_output_dir(task_num, utilization_rate)
    results_path = os.path.join(output_dir, "evaluation_monte_carlo_adjust_sample.csv")
    existing_results = load_existing_results(results_path, ["TaskSetID", "TaskNum", "UtilizationRate", "Samples", "WCDFP", "ExecutionTime", "DeadlineMissCount"])

    taskset = TaskSet(task_num=task_num, utilization_rate=utilization_rate, seed=taskset_id)
    absolute_deadline = taskset.timeline[0][-1].absolute_deadline

    for samples in sample_sizes:
        if samples in existing_results["Samples"].values:
            print(f"Samples {samples} already evaluated. Skipping.")
            continue

        print(f"Evaluating with {samples} samples...")
        start_time = time.time()
        response_times_mc, wcdfp_mc = calculate_response_time_distribution(
            taskset=taskset,
            target_job=taskset.target_job,
            false_probability=false_probability,
            thread_num=thread_num,
            log_flag=log_flag,
            plot_flag=plot_flag,
            samples=samples,
            seed=samples,
        )
        elapsed_time = time.time() - start_time
        miss_count = count_deadline_misses(response_times_mc, absolute_deadline)
        result = {
            "TaskSetID": taskset_id,
            "TaskNum": task_num,
            "UtilizationRate": utilization_rate,
            "Samples": samples,
            "WCDFP": wcdfp_mc,
            "ExecutionTime": elapsed_time,
            "DeadlineMissCount": miss_count,
        }
        append_result(results_path, result)
        print(f"Samples {samples} evaluation completed.")

    print(f"Monte Carlo adjust sample results saved to {results_path}")


def evaluate_berry_essen(
    task_num=default_task_num,
    utilization_rate=default_utilization_rate,
    total_taskset=default_total_taskset,
    log_flag=default_log_flag,
):
    output_dir = get_output_dir(task_num, utilization_rate)
    results_path = os.path.join(output_dir, "evaluation_berry_essen.csv")
    existing_results = load_existing_results(results_path, ["TaskSetID", "TaskNum", "UtilizationRate", "WCDFP", "ExecutionTime"])
    evaluated_ids = set(existing_results["TaskSetID"].values)

    with tqdm(total=total_taskset, desc="Evaluating Berry-Essen", unit="set") as progress_bar:
        for i in range(total_taskset):
            taskset_id = i + 1
            if taskset_id in evaluated_ids:
                progress_bar.update(1)
                continue

            progress_bar.set_description(f"Evaluating TaskSet {taskset_id}/{total_taskset}")
            taskset = TaskSet(task_num=task_num, utilization_rate=utilization_rate, seed=taskset_id)
            start_time = time.time()
            wcdfp_be, _ = calculate_response_time_by_berry_essen(
                taskset=taskset,
                target_job=taskset.target_job,
                A=A,
                upper=True,
                log_flag=log_flag,
                seed=i,
            )
            elapsed_time = time.time() - start_time

            result = {
                "TaskSetID": taskset_id,
                "TaskNum": task_num,
                "UtilizationRate": utilization_rate,
                "WCDFP": wcdfp_be,
                "ExecutionTime": elapsed_time,
            }
            append_result(results_path, result)
            progress_bar.update(1)

    print(f"Berry-Essen results saved to {results_path}")


def evaluate_convolution_generic(evaluation_fn, file_label, task_num, utilization_rate, total_taskset, thread_num, log_flag, float128_flag):
    output_dir = get_output_dir(task_num, utilization_rate)
    file_name = f"evaluation_{file_label}" + ("_float128.csv" if float128_flag else ".csv")
    results_path = os.path.join(output_dir, file_name)
    columns = ["TaskSetID", "TaskNum", "UtilizationRate", "WCDFP", "ExecutionTime"]
    # Some methods include DeadlineMissCount; add if needed.
    if file_label in ("monte_carlo", "convolution_merge", "evaluation_convolution_doubling"):
        columns.append("DeadlineMissCount")
    existing_results = load_existing_results(results_path, columns)
    evaluated_ids = set(existing_results["TaskSetID"].values)

    with tqdm(total=total_taskset, desc=f"Evaluating {file_label}", unit="set") as progress_bar:
        for i in range(total_taskset):
            taskset_id = i + 1
            if taskset_id in evaluated_ids:
                progress_bar.update(1)
                continue

            seed = i * thread_num
            progress_bar.set_description(f"Evaluating TaskSet {taskset_id}/{total_taskset}")
            taskset = TaskSet(task_num=task_num, utilization_rate=utilization_rate, seed=taskset_id)
            start_time = time.time()
            # evaluation_fn should return (wcdfp, optional deadline miss count)
            ret = evaluation_fn(taskset, seed, log_flag, float128_flag)
            elapsed_time = time.time() - start_time
            if isinstance(ret, tuple) and len(ret) == 2:
                wcdfp, miss_count = ret
            else:
                wcdfp, miss_count = ret, None

            result = {
                "TaskSetID": taskset_id,
                "TaskNum": task_num,
                "UtilizationRate": utilization_rate,
                "WCDFP": wcdfp,
                "ExecutionTime": elapsed_time,
            }
            if miss_count is not None:
                result["DeadlineMissCount"] = miss_count

            append_result(results_path, result)
            progress_bar.update(1)

    print(f"{file_label.capitalize()} results saved to {results_path}")


def eval_conv_doubling(taskset, seed, log_flag, float128_flag):
    _, wcdfp = calculate_response_time_with_doubling(
        taskset=taskset,
        target_job=taskset.target_job,
        log_flag=log_flag,
        float128_flag=float128_flag,
    )
    # Deadline misses are not computed here.
    return wcdfp


def eval_conv(taskset, seed, log_flag, float128_flag):
    _, wcdfp = calculate_response_time_by_conv(
        taskset=taskset,
        target_job=taskset.target_job,
        log_flag=log_flag,
        float128_flag=float128_flag,
    )
    return wcdfp


def eval_conv_merge(taskset, seed, log_flag, float128_flag):
    _, wcdfp = calculate_response_time_with_merge(
        taskset=taskset,
        target_job=taskset.target_job,
        log_flag=log_flag,
        float128_flag=float128_flag,
    )
    return wcdfp


def evaluate_convolution_doubling(task_num=default_task_num, utilization_rate=default_utilization_rate,
                                  total_taskset=default_total_taskset, thread_num=default_thread_num,
                                  log_flag=default_log_flag, float128_flag=default_float128_flag):
    # Use the generic evaluator helper
    evaluate_convolution_generic(eval_conv_doubling, "convolution_doubling", task_num, utilization_rate,
                                 total_taskset, thread_num, log_flag, float128_flag)


def evaluate_convolution(task_num=default_task_num, utilization_rate=default_utilization_rate,
                         total_taskset=default_total_taskset, thread_num=default_thread_num,
                         log_flag=default_log_flag, float128_flag=default_float128_flag):
    evaluate_convolution_generic(eval_conv, "convolution", task_num, utilization_rate,
                                 total_taskset, thread_num, log_flag, float128_flag)


def evaluate_convolution_merge(task_num=default_task_num, utilization_rate=default_utilization_rate,
                               total_taskset=default_total_taskset, thread_num=default_thread_num,
                               log_flag=default_log_flag, float128_flag=default_float128_flag):
    evaluate_convolution_generic(eval_conv_merge, "convolution_merge", task_num, utilization_rate,
                                 total_taskset, thread_num, log_flag, float128_flag)


def evaluate_all_methods():
    all_task_num = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    all_utilization_rate = [0.60, 0.65, 0.70]
    for t_num in all_task_num:
        for u_rate in all_utilization_rate:
            print(f"Evaluating for TaskNum: {t_num}, UtilizationRate: {u_rate}")
            evaluate_monte_carlo(task_num=t_num, utilization_rate=u_rate)
            evaluate_berry_essen(task_num=t_num, utilization_rate=u_rate)
            evaluate_convolution_doubling(task_num=t_num, utilization_rate=u_rate)
            evaluate_convolution_merge(task_num=t_num, utilization_rate=u_rate)
            evaluate_convolution(task_num=t_num, utilization_rate=u_rate)


def plot_normalized_response_times(
    task_num=30,
    utilization_rate=0.60,
    false_probability=1e-6,
    error_margin=0.01,
    float128_flag=default_float128_flag,
):
    taskset_seed = 16
    taskset = TaskSet(task_num=task_num, utilization_rate=utilization_rate, seed=taskset_seed)
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
    response_time_conv = np.asarray(response_time_conv)
    cdf_conv = np.cumsum(response_time_conv) / np.sum(response_time_conv)
    indices_conv = np.arange(len(response_time_conv)) * MINIMUM_TIME_UNIT
    methods_data.append(("Convolution", indices_conv, cdf_conv))
    print(f"Convolution evaluation completed in {time.time() - start_time:.2f} seconds.")

    # Doubling Convolution
    start_time = time.time()
    response_time_doubling, wcdfp_doubling = calculate_response_time_with_doubling(
        taskset,
        taskset.target_job,
        float128_flag=float128_flag
    )
    response_time_doubling = np.concatenate([response_time_doubling, [wcdfp_doubling]])
    response_time_doubling = np.asarray(response_time_doubling)
    cdf_doubling = np.cumsum(response_time_doubling) / np.sum(response_time_doubling)
    indices_doubling = np.arange(len(response_time_doubling)) * MINIMUM_TIME_UNIT
    methods_data.append(("Doubling", indices_doubling, cdf_doubling))
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

    plt.figure(figsize=(12, 8))
    for label, times, cdf in methods_data:
        plt.step(times, cdf, label=label, where="post")

    deadline = taskset.target_job.absolute_deadline
    plt.axvline(x=deadline, color="red", linestyle="--", label="Deadline")
    plt.xlabel("Normalized Response Time")
    plt.ylabel("Cumulative Distribution Function")
    plt.title("Normalized Response Time CDFs Across Methods")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    output_path = os.path.join(output_prefix, "response_time_cdfs_normalized.png")
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")
