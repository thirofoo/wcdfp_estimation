import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import norm, beta
from concurrent.futures import ProcessPoolExecutor
from common.parameters import MINIMUM_TIME_UNIT


def required_sample_size_binomial(error_margin, false_probability):
    """
    Calculate the required sample size for a binomial distribution using a normal approximation.

    Parameters:
    - error_margin: Allowed margin of error (epsilon).
    - false_probability: Probability of a false result (alpha level).
    - output_dir: Directory to save the output log file.

    Returns:
    - required_n: Required sample size.
    """
    # Calculate required sample size
    z = norm.ppf(1 - (false_probability / 2))  # z-score (confidence interval)
    p = 0.5  # Maximum uncertainty for the deadline miss probability (worst case)
    required_n = (z**2 * p * (1 - p)) / (error_margin**2)
    return np.ceil(required_n)


def calculate_response_time(taskset, target_job, log_flag=False, traditional_ci=False):
    """
    Calculate the response time of a target job using a timeline simulation.

    :param taskset: TaskSet containing tasks and timeline
    :param target_job: The target job whose response time is to be calculated
    :param log_flag: If True, logs detailed intermediate results
    :param traditional_ci: If True, ignores carry-in at time 0
    :return: Calculated response time of the target job
    """
    carry_in = 0
    boundary = 0
    total_prev_cost = 0  # Total previous cost up to the boundary
    arrival_times = taskset.arrival_times  # Get arrival times from the taskset
    timeline = taskset.timeline  # Get timeline from the taskset

    # Determine the boundary for the timeline, which is the first arrival time
    # after the target job's release time
    while boundary < len(arrival_times) and arrival_times[boundary] * MINIMUM_TIME_UNIT < target_job.release_time:
        boundary += 1

    # Calculate initial carry-in at time t = 0
    carry_in = target_job.task.get_execution_time()
    for job in timeline[arrival_times[0]]:
        # Only consider jobs with priority higher or equal to the target job
        if job < target_job:
            break
        carry_in += job.task.get_execution_time()

    # If traditional carry-in mode is enabled, set carry-in to 0
    if traditional_ci:
        carry_in = 0

    if log_flag:
        print(f"carry_in: {carry_in}")
        print(f"total_prev_cost: {total_prev_cost}")
        print(f"boundary: {boundary}")

    # Start calculating response time from the boundary onward
    idx = boundary
    total_next_cost = 0  # Cost from future jobs
    while idx < len(arrival_times) and arrival_times[idx] * MINIMUM_TIME_UNIT <= target_job.absolute_deadline:
        # Iterate through all jobs arriving at the current timeline index
        for job in timeline[arrival_times[idx]]:
            if job < target_job or job == target_job:
                continue  # Skip jobs with lower priority or the same job
            total_next_cost += job.task.get_execution_time()

        # If the carry-in and next cost fit within the current interval, return response time
        if carry_in + total_next_cost <= arrival_times[idx] * MINIMUM_TIME_UNIT - arrival_times[boundary] * MINIMUM_TIME_UNIT:
            if log_flag:
                print("\n========== Scheduling Succeeded !! ==========\n")
            return carry_in + total_next_cost

        idx += 1  # Move to the next arrival time

    # Check if the response time fits within the absolute deadline
    if carry_in + total_next_cost <= target_job.absolute_deadline - arrival_times[boundary] * MINIMUM_TIME_UNIT:
        if log_flag:
            print("\n========== Scheduling Succeeded !! ==========\n")
        return carry_in + total_next_cost

    # If the deadline is missed, return the exact response time
    if log_flag:
        print("\n========== Scheduling Failed ... ==========\n")
    return carry_in + total_next_cost


def calculate_deadline_miss_probability_agresti_coull(deadline_miss_cnt, total_samples, false_probability):
    """
    Calculate the upper bound of deadline miss probability using Agresti-Coull interval.

    :param deadline_miss_cnt: Count of deadline misses
    :param total_samples: Total number of samples
    :param false_probability: Significance level (1 - confidence level)
    :return: Upper bound of WCDFP
    """
    z = norm.ppf(1 - (false_probability / 2))
    s_tilde = total_samples + z**2
    p_tilde = (deadline_miss_cnt + z**2 / 2) / s_tilde
    upper_bound = p_tilde + z * np.sqrt(p_tilde * (1 - p_tilde) / s_tilde)
    return upper_bound


def calculate_deadline_miss_probability_jeffreys(deadline_miss_cnt, total_samples, false_probability):
    """
    Calculate the upper bound of deadline miss probability using Jeffreys interval.

    :param deadline_miss_cnt: Count of deadline misses
    :param total_samples: Total number of samples
    :param false_probability: Significance level (1 - confidence level)
    :return: Upper bound of WCDFP
    """
    return beta.ppf(1 - false_probability / 2, deadline_miss_cnt + 0.5, total_samples - deadline_miss_cnt + 0.5)


def sample_responses(n_samples, taskset, target_job, seed):
    """
    Generate response time samples and count deadline misses for parallel processing.

    :param n_samples: Number of samples to generate
    :param taskset: TaskSet containing tasks
    :param target_job: Target job for response time sampling
    :param seed: Random seed for reproducibility
    :return: List of response times and count of deadline misses
    """
    np.random.seed(seed)
    sub_response_times = []
    sub_deadline_miss_cnt = 0

    for _ in range(n_samples):
        response_time = calculate_response_time(taskset, target_job)
        sub_response_times.append(response_time)
        if response_time >= target_job.absolute_deadline:
            sub_deadline_miss_cnt += 1
    return sub_response_times, sub_deadline_miss_cnt


def calculate_response_time_distribution(taskset, target_job, false_probability=0.000001,
                                         log_flag=False, traditional_ci=False, plot_flag=False, seed=0, thread_num=1, samples=0):
    """
    Calculate the distribution of response times for a given target job in the taskset.

    :param taskset: TaskSet containing tasks and timeline
    :param target_job: Target job for which response times are calculated
    :param false_probability: False positive rate for confidence interval
    :param log_flag: If True, logs detailed intermediate results and shows tqdm progress bar
    :param traditional_ci: If True, uses traditional carry-in calculations
    :param plot_flag: If True, plots a histogram of response times
    :param seed: Random seed for reproducibility
    :param thread_num: Number of threads to use for parallel processing
    :param samples: Total number of response time samples to generate
    :return: Tuple of response_times and deadline_miss_prob_upper_jeffreys
    """
    np.random.seed(seed)  # Set the random seed for reproducibility

    # Initialize list to store response times and a counter for deadline misses
    response_times = []
    deadline_miss_cnt = 0

    # Calculate the number of samples per thread
    samples_per_thread = int(samples // thread_num)

    if thread_num == 1:
        # Single-threaded processing
        sample_range = tqdm(range(int(samples)), desc="Collecting Samples") if log_flag else range(int(samples))
        for _ in sample_range:
            response_time = calculate_response_time(taskset, target_job, log_flag=log_flag, traditional_ci=traditional_ci)
            response_times.append(response_time)
            if response_time >= target_job.absolute_deadline:
                deadline_miss_cnt += 1
    else:
        # Multi-threaded processing using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=thread_num) as executor:
            futures = []
            for i in range(thread_num):
                # Calculate number of samples for each thread (distribute remainders evenly)
                n_samples = samples_per_thread + (1 if i < int(samples % thread_num) else 0)
                thread_seed = seed + i  # Ensure unique seed for each thread
                futures.append(executor.submit(sample_responses, n_samples, taskset, target_job, thread_seed))

            # Collect results from each thread
            process_range = tqdm(futures, desc="Collecting Results from Processes") if log_flag else futures
            for future in process_range:
                sub_response_times, sub_deadline_miss_cnt = future.result()
                response_times.extend(sub_response_times)
                deadline_miss_cnt += sub_deadline_miss_cnt

    # Calculate deadline miss probability upper bound using Agresti-Coull interval
    deadline_miss_prob_upper_agresti_coull = calculate_deadline_miss_probability_agresti_coull(
        deadline_miss_cnt, samples, false_probability
    )

    # Calculate deadline miss probability upper bound using Jeffreys interval
    deadline_miss_prob_upper_jeffreys = calculate_deadline_miss_probability_jeffreys(
        deadline_miss_cnt, samples, false_probability
    )

    # Log results if log_flag is True
    if log_flag:
        print("【WCDFP (agresti_coull)】: ", deadline_miss_prob_upper_agresti_coull)
        print("【WCDFP (jeffreys)】: ", deadline_miss_prob_upper_jeffreys)
        print(f"Deadline Miss Count: {deadline_miss_cnt}")
        print(f"DFP: {deadline_miss_cnt / samples}")
        print(f"Total Response Times Collected: {len(response_times)}")

    # Plot histogram of response times if plot_flag is True
    if plot_flag:
        plt.figure(figsize=(10, 6))
        plt.hist(response_times, bins=100, color='blue', alpha=0.7)
        plt.title(f"Response Time Distribution (Sampled {int(samples)} times)")
        plt.xlabel("Response Time")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()

    # return response_times, deadline_miss_prob_upper_jeffreys
    return response_times, deadline_miss_prob_upper_agresti_coull
