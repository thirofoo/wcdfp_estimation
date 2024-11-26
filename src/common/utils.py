import math
from scipy.stats import truncnorm

from common.parameters import MINIMUM_TIME_UNIT

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
