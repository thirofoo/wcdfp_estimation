import numpy as np
from scipy.stats import norm
from tqdm import tqdm
from common.parameters import ABNORMAL_MODE_PROB


def calculate_response_time_by_berry_essen(taskset, target_job, A=1.0, upper=True, log_flag=False, seed=0):
    """
    Calculate WCDFP using Berry-Essen theorem.

    :param taskset: TaskSet containing tasks
    :param target_job: Target job for which WCDFP is calculated
    :param A: Penalty coefficient
    :param upper: If True, calculate upper bound; otherwise, calculate lower bound
    :param log_flag: If True, logs detailed intermediate results
    :return: WCDFP and (x_values, berry_essen_cdf_values)
    :param seed: Random seed for reproducibility
    """
    np.random.seed(seed)  # Set the random seed for reproducibility

    absolute_deadline = target_job.absolute_deadline
    rho_sum = 0
    alpha_sigma_sum = 0
    mu_sum = 0
    max_sum = 0

    for task in tqdm(taskset.tasks, desc="Processing tasks", disable=not log_flag):
        alpha_i_t = np.ceil((absolute_deadline + task.relative_deadline) / task.minimum_inter_arrival_time)
        normal_mu = task.normal_mu
        abnormal_mu = task.abnormal_mu
        normal_sigma = task.normal_sigma
        abnormal_sigma = task.abnormal_sigma

        expected_mu = (1 -ABNORMAL_MODE_PROB) * normal_mu +ABNORMAL_MODE_PROB * abnormal_mu
        expected_variance = (
            (1 -ABNORMAL_MODE_PROB) * normal_sigma**2
            +ABNORMAL_MODE_PROB * abnormal_sigma**2
            + (1 -ABNORMAL_MODE_PROB) * (normal_mu - expected_mu) ** 2
            +ABNORMAL_MODE_PROB * (abnormal_mu - expected_mu) ** 2
        )

        mu_sum += alpha_i_t * expected_mu
        alpha_sigma_sum += alpha_i_t**2 * expected_variance
        max_sum += alpha_i_t * task.wcet

        execution_time_samples = np.array([task.get_execution_time() for _ in range(1000)])
        rho_i = alpha_i_t * np.mean(np.abs(execution_time_samples - expected_mu) ** 3)
        rho_sum += rho_i

    sigma_total = np.sqrt(alpha_sigma_sum)
    psi = (sigma_total**-3) * rho_sum

    if log_flag:
        print(f"Penalty term ψ: {psi}")
        print(f"Sum of means (μ): {mu_sum}")
        print(f"Total variance: {alpha_sigma_sum}")
        print(f"Maximum response time (max_sum): {max_sum}")

    x_values = np.linspace(mu_sum - 4 * sigma_total, mu_sum + 4 * sigma_total, 1000)

    if upper:
        berry_essen_cdf_values = norm.cdf((x_values - mu_sum) / sigma_total) - A * psi
    else:
        berry_essen_cdf_values = norm.cdf((x_values - mu_sum) / sigma_total) + A * psi

    berry_essen_cdf_values = np.clip(berry_essen_cdf_values, 0, 1)
    berry_essen_cdf_values[x_values >= max_sum] = 1

    # Calculate WCDFP as 1 - CDF(deadline)
    wcdfp = 1 - berry_essen_cdf_values[np.searchsorted(x_values, absolute_deadline, side="right") - 1]

    return wcdfp, (x_values, berry_essen_cdf_values)
