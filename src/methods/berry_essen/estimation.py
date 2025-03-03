import numpy as np
from scipy.stats import norm
from tqdm import tqdm
from common.parameters import ABNORMAL_MODE_PROB, BERRY_ESSEN_COEFFICIENT


def calc_expected_stats(task):
    """
    Calculate expected mean and variance for the task considering normal and abnormal modes.
    """
    mu = (1 - ABNORMAL_MODE_PROB) * task.normal_mu + ABNORMAL_MODE_PROB * task.abnormal_mu
    variance = (
        (1 - ABNORMAL_MODE_PROB) * task.normal_sigma**2
        + ABNORMAL_MODE_PROB * task.abnormal_sigma**2
        + (1 - ABNORMAL_MODE_PROB) * (task.normal_mu - mu) ** 2
        + ABNORMAL_MODE_PROB * (task.abnormal_mu - mu) ** 2
    )
    return mu, variance


def calculate_wcdfp_by_berry_essen(taskset, target_job, A=BERRY_ESSEN_COEFFICIENT, upper=True, log_flag=False, seed=0):
    """
    Calculate WCDFP using the Berry-Essen theorem.
    """
    np.random.seed(seed)
    abs_deadline = target_job.absolute_deadline

    total_mu = 0.0
    total_variance = 0.0
    total_rho = 0.0
    max_response_time = 0.0

    # Process each task from the taskset.
    for task in tqdm(taskset.tasks, desc="Processing tasks", disable=not log_flag):
        # Compute the number of job instances for this task.
        job_count = np.ceil((abs_deadline + task.relative_deadline) / task.minimum_inter_arrival_time)
        mu, variance = calc_expected_stats(task)

        total_mu += job_count * mu
        total_variance += job_count**2 * variance
        max_response_time += job_count * task.wcet

        # Compute the third central moment from 1000 samples of execution times.
        samples = np.array([task.get_execution_time() for _ in range(1000)])
        total_rho += job_count * np.mean(np.abs(samples - mu) ** 3)

    sigma_total = np.sqrt(total_variance)
    berry_essen_penalty = total_rho / (sigma_total**3) if sigma_total != 0 else 0

    if log_flag:
        print(f"Penalty term ψ: {berry_essen_penalty}")
        print(f"Total mean (μ): {total_mu}")
        print(f"Total variance: {total_variance}")
        print(f"Maximum response time: {max_response_time}")

    # Define range for evaluating the CDF.
    x_min = total_mu - 4 * sigma_total
    x_max = total_mu + 4 * sigma_total
    x_values = np.linspace(x_min, x_max, 1000)
    standardized_x = (x_values - total_mu) / sigma_total
    base_cdf = norm.cdf(standardized_x)

    # Adjust the CDF using Berry-Essen penalty.
    adjustment = A * berry_essen_penalty
    if upper:
        berry_essen_cdf_values = base_cdf - adjustment
    else:
        berry_essen_cdf_values = base_cdf + adjustment

    berry_essen_cdf_values = np.clip(berry_essen_cdf_values, 0, 1)
    berry_essen_cdf_values[x_values >= max_response_time] = 1

    # WCDFP is 1 minus the CDF evaluated at the deadline.
    deadline_idx = np.searchsorted(x_values, abs_deadline, side="right") - 1
    wcdfp = 1 - berry_essen_cdf_values[deadline_idx]

    return wcdfp, (x_values, berry_essen_cdf_values)
