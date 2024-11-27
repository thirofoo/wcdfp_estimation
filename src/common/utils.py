import math
import numpy as np
from scipy.stats import truncnorm

from common.parameters import (
    MINIMUM_TIME_UNIT,
    ABNORMAL_MODE_PROB,
    NORMAL_MEAN_COEFF,
    NORMAL_STD_COEFF,
    ABNORMAL_MEAN_COEFF,
    ABNORMAL_STD_COEFF,
    SPARSITY_THRESHOLD,
)

# ==================== Helper Functions ==================== #

def round_min_unit(time):
    """
    Round the given time to the nearest multiple of MINIMUM_TIME_UNIT.
    """
    return math.ceil(time / MINIMUM_TIME_UNIT) * MINIMUM_TIME_UNIT


def get_truncated_normal(mu, sigma, lower_bound, upper_bound):
    """
    Return a truncated normal distribution within [lower_bound, upper_bound].
    """
    a, b = (lower_bound - mu) / sigma, (upper_bound - mu) / sigma
    return truncnorm(a, b, loc=mu, scale=sigma)


def calculate_truncated_mean(mu, sigma, lower_bound, upper_bound):
    """
    Calculate the mean of a truncated normal distribution.
    """
    a, b = (lower_bound - mu) / sigma, (upper_bound - mu) / sigma
    return truncnorm.mean(a, b, loc=mu, scale=sigma)

# ==================== WCET Calculation ==================== #

def calculate_expected_execution_time(wcet, trunc_lower, trunc_upper):
    """
    Calculate the expected execution time within the specified range.
    """
    range_size = (trunc_upper - trunc_lower) / MINIMUM_TIME_UNIT

    if range_size <= SPARSITY_THRESHOLD:
        # If the range is sparse, discretize time intervals
        discrete_times = np.arange(trunc_lower, trunc_upper + MINIMUM_TIME_UNIT, MINIMUM_TIME_UNIT)
        expected_execution_time = 0.0

        # Compute the expected execution time by iterating over discrete values
        for t in discrete_times:
            probability = (
                (1 - ABNORMAL_MODE_PROB) * get_truncated_normal(
                    wcet / NORMAL_MEAN_COEFF, wcet / NORMAL_STD_COEFF, 0, wcet
                ).pdf(t) +
                ABNORMAL_MODE_PROB * get_truncated_normal(
                    wcet / ABNORMAL_MEAN_COEFF, wcet / ABNORMAL_STD_COEFF, 0, wcet
                ).pdf(t)
            )
            expected_execution_time += probability * MINIMUM_TIME_UNIT * t
    else:
        # Use normal distribution approximation if the range is dense
        normal_mu = wcet / NORMAL_MEAN_COEFF
        normal_sigma = wcet / NORMAL_STD_COEFF
        abnormal_mu = wcet / ABNORMAL_MEAN_COEFF
        abnormal_sigma = wcet / ABNORMAL_STD_COEFF

        normal_truncated_mean = calculate_truncated_mean(normal_mu, normal_sigma, trunc_lower, trunc_upper)
        abnormal_truncated_mean = calculate_truncated_mean(abnormal_mu, abnormal_sigma, trunc_lower, trunc_upper)

        expected_execution_time = (
            (1 - ABNORMAL_MODE_PROB) * normal_truncated_mean +
            ABNORMAL_MODE_PROB * abnormal_truncated_mean
        )

    return expected_execution_time


def calculate_wcet(minimum_inter_arrival_time, rate, tol=1e-10):
    """
    Calculate the WCET (Worst-Case Execution Time) for a given task.
    """
    low, high = 0, minimum_inter_arrival_time
    target = minimum_inter_arrival_time * rate

    # Perform binary search to find WCET
    while high - low > tol:
        mid = (low + high) / 2.0
        expected_time = calculate_expected_execution_time(mid, 0, mid)
        if expected_time < target:
            low = mid
        else:
            high = mid

    return (low + high) / 2.0
