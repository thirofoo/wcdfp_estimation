import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import norm, beta
from concurrent.futures import ProcessPoolExecutor
from common.parameters import MINIMUM_TIME_UNIT


def required_sample_size_binomial(error_margin, false_probability):
    """
    Calculate the required sample size for a binomial distribution.
    """
    z = norm.ppf(1 - false_probability / 2)  # z-score for given confidence level
    p = 0.5  # maximum uncertainty (worst case)
    required_n = (z ** 2 * p * (1 - p)) / (error_margin ** 2)
    return np.ceil(required_n)


def calculate_response_time_by_monte_carlo(taskset, target_job, log_flag=False):
    """
    Compute the response time of a target job using a timeline simulation.
    """
    carry_in = 0
    boundary = 0
    total_prev_cost = 0
    arrival_times = taskset.arrival_times
    timeline = taskset.timeline

    # Determine the first index with arrival time after target job's release time
    while boundary < len(arrival_times) and arrival_times[boundary] * MINIMUM_TIME_UNIT < target_job.release_time:
        boundary += 1

    # Initialize carry-in for all tasks
    for task in taskset.tasks:
        release_count = task.relative_deadline // task.minimum_inter_arrival_time
        if target_job.task == task:
            release_count = 1
        while release_count > 0:
            carry_in += task.get_execution_time()
            release_count -= 1

    if log_flag:
        print(f"carry_in: {carry_in}")
        print(f"total_prev_cost: {total_prev_cost}")
        print(f"boundary: {boundary}")

    idx = boundary
    total_next_cost = 0

    # Process future jobs until target job's deadline
    while idx < len(arrival_times) and arrival_times[idx] * MINIMUM_TIME_UNIT <= target_job.absolute_deadline:
        # Check if total cost fits within the interval
        if carry_in + total_next_cost <= (arrival_times[idx] - arrival_times[boundary]) * MINIMUM_TIME_UNIT:
            if log_flag:
                print("\n========== Scheduling Succeeded !! ==========\n")
            return carry_in + total_next_cost

        # Add execution times for jobs at the current time slot if priority is higher
        for job in timeline[arrival_times[idx]]:
            if job < target_job or job == target_job:
                continue
            total_next_cost += job.task.get_execution_time()

        idx += 1

    # If the accumulated cost fits within the absolute deadline
    if carry_in + total_next_cost <= (target_job.absolute_deadline - arrival_times[boundary] * MINIMUM_TIME_UNIT):
        if log_flag:
            print("\n========== Scheduling Succeeded !! ==========\n")
        return carry_in + total_next_cost

    if log_flag:
        print("\n========== Scheduling Failed ... ==========\n")
    return carry_in + total_next_cost


def calculate_deadline_miss_probability_agresti_coull(deadline_miss_cnt, total_samples, false_probability):
    """
    Compute the upper bound using the Agresti-Coull interval.
    """
    z = norm.ppf(1 - false_probability / 2)
    s_tilde = total_samples + z ** 2
    p_tilde = (deadline_miss_cnt + z ** 2 / 2) / s_tilde
    upper_bound = p_tilde + z * np.sqrt(p_tilde * (1 - p_tilde) / s_tilde)
    return upper_bound


def calculate_deadline_miss_probability_jeffreys(deadline_miss_cnt, total_samples, false_probability):
    """
    Compute the upper bound using the Jeffreys interval.
    """
    return beta.ppf(1 - false_probability / 2, deadline_miss_cnt + 0.5, total_samples - deadline_miss_cnt + 0.5)


def sample_responses(n_samples, taskset, target_job, seed):
    """
    Generate response time samples and count deadline misses.
    """
    np.random.seed(seed)
    sub_response_times = []
    sub_deadline_miss_cnt = 0

    for _ in range(int(n_samples)):
        response_time = calculate_response_time_by_monte_carlo(taskset, target_job)
        sub_response_times.append(response_time)
        if response_time >= target_job.absolute_deadline:
            sub_deadline_miss_cnt += 1

    return sub_response_times, sub_deadline_miss_cnt


def calculate_wcdfp_by_monte_carlo(taskset, target_job, false_probability=0.000001,
                                   log_flag=False, plot_flag=False, seed=0, thread_num=1, samples=0):
    """
    Calculate the response time distribution for a target job using Monte Carlo simulation.
    """
    np.random.seed(seed)  # ensure reproducibility
    response_times = []
    deadline_miss_cnt = 0

    samples_per_thread = samples // thread_num

    if thread_num == 1:
        sample_iter = tqdm(range(samples), desc="Collecting Samples") if log_flag else range(samples)
        for _ in sample_iter:
            response_time = calculate_response_time_by_monte_carlo(taskset, target_job, log_flag=log_flag)
            response_times.append(response_time)
            if response_time >= target_job.absolute_deadline:
                deadline_miss_cnt += 1
    else:
        with ProcessPoolExecutor(max_workers=thread_num) as executor:
            futures = []
            for i in range(thread_num):
                n_samples = samples_per_thread + (1 if i < samples % thread_num else 0)
                thread_seed = seed + i
                futures.append(executor.submit(sample_responses, n_samples, taskset, target_job, thread_seed))

            process_iter = tqdm(futures, desc="Collecting Results from Processes") if log_flag else futures
            for future in process_iter:
                sub_response_times, sub_deadline_miss_cnt = future.result()
                response_times.extend(sub_response_times)
                deadline_miss_cnt += sub_deadline_miss_cnt

    # Calculate deadline miss probability using Agresti-Coull interval
    deadline_miss_prob_upper_agresti_coull = calculate_deadline_miss_probability_agresti_coull(
        deadline_miss_cnt, samples, false_probability
    )

    # Calculate deadline miss probability using Jeffreys interval
    deadline_miss_prob_upper_jeffreys = calculate_deadline_miss_probability_jeffreys(
        deadline_miss_cnt, samples, false_probability
    )

    if log_flag:
        print("WCDFP (Agresti-Coull):", deadline_miss_prob_upper_agresti_coull)
        print("WCDFP (Jeffreys):", deadline_miss_prob_upper_jeffreys)
        print(f"Deadline Miss Count: {deadline_miss_cnt}")
        print(f"DFP: {deadline_miss_cnt / samples}")
        print(f"Total Response Times Collected: {len(response_times)}")

    if plot_flag:
        plt.figure(figsize=(10, 6))
        plt.hist(response_times, bins=100, color='blue', alpha=0.7)
        plt.title(f"Response Time Distribution (Sampled {samples} times)")
        plt.xlabel("Response Time")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()

    return response_times, deadline_miss_prob_upper_agresti_coull
