import numpy as np
from bisect import bisect_left
from scipy.integrate import quad
from common.utils import get_truncated_normal, round_min_unit
from common.parameters import (
    MINIMUM_TIME_UNIT, ABNORMAL_MODE_PROB,
    NORMAL_MEAN_COEFF, NORMAL_STD_COEFF,
    ABNORMAL_MEAN_COEFF, ABNORMAL_STD_COEFF
)


class Task:
    """
    Represents a periodic task in the scheduling system.
    Includes execution time modeling and weighted sampling.
    """
    def __init__(self, wcet, relative_deadline, minimum_inter_arrival_time, theta=0.5):
        """
        Initialize the Task with WCET, deadline, and execution time distributions.
        :param wcet: Worst-Case Execution Time
        :param relative_deadline: Relative deadline of the task
        :param minimum_inter_arrival_time: Minimum time between consecutive task activations
        :param theta: Parameter for tilted distribution
        """
        self.wcet = round_min_unit(wcet)
        self.relative_deadline = relative_deadline
        self.minimum_inter_arrival_time = round_min_unit(minimum_inter_arrival_time)
        self.theta = theta

        # Normal and abnormal execution time distributions
        self.normal_mu = self.wcet / NORMAL_MEAN_COEFF
        self.normal_sigma = self.wcet / NORMAL_STD_COEFF
        self.normal_dist = get_truncated_normal(
            mu=self.normal_mu, sigma=self.normal_sigma, lower_bound=0, upper_bound=self.wcet
        )

        self.abnormal_mu = self.wcet / ABNORMAL_MEAN_COEFF
        self.abnormal_sigma = self.wcet / ABNORMAL_STD_COEFF
        self.abnormal_dist = get_truncated_normal(
            mu=self.abnormal_mu, sigma=self.abnormal_sigma, lower_bound=0, upper_bound=self.wcet
        )

        # Generate cumulative distribution functions
        self.create_cdf(self.theta)

    def create_cdf(self, theta):
        """
        Create cumulative distribution functions (CDFs) for the task.
        This includes original and tilted distributions for weighted sampling.
        """
        # Define value range based on the WCET
        min_value = 0
        max_value = self.wcet
        theta /= self.wcet  # Normalize theta

        adjusted_min_value = max(MINIMUM_TIME_UNIT, np.ceil(min_value / MINIMUM_TIME_UNIT) * MINIMUM_TIME_UNIT)
        adjusted_max_value = np.floor(max_value / MINIMUM_TIME_UNIT) * MINIMUM_TIME_UNIT

        # Create discrete values within the range
        self.x_values = np.arange(adjusted_min_value, adjusted_max_value + MINIMUM_TIME_UNIT, MINIMUM_TIME_UNIT)
        if self.x_values[0] != 0:
            self.x_values = np.insert(self.x_values, 0, 0.0)

        # Compute original PDF
        normal_pdf_values = (1 - ABNORMAL_MODE_PROB) * self.normal_dist.pdf(self.x_values)
        abnormal_pdf_values = ABNORMAL_MODE_PROB * self.abnormal_dist.pdf(self.x_values)
        self.original_pdf_values = normal_pdf_values + abnormal_pdf_values
        self.original_pdf_values[0] = 0.0  # Set probability at 0 to 0

        # Normalize the original PDF using the trapezoidal rule
        normalization_constant_original = np.trapz(self.original_pdf_values, self.x_values)
        self.original_pdf_values /= normalization_constant_original

        # Compute original CDF
        self.cdf_values = np.cumsum(self.original_pdf_values)
        self.cdf_values /= self.cdf_values[-1]  # Ensure the last value is 1

        # Compute tilted PDF in log-space
        epsilon = 1e-10
        log_original_pdf_values = np.log(np.maximum(self.original_pdf_values, epsilon))
        log_tilted_pdf_values = log_original_pdf_values + theta * self.x_values
        self.tilted_pdf_values = np.exp(log_tilted_pdf_values)

        # Normalize tilted PDF using the trapezoidal rule
        normalization_constant_tilted = np.trapz(self.tilted_pdf_values, self.x_values)
        self.tilted_pdf_values /= normalization_constant_tilted

        # Compute tilted CDF
        self.tilted_cdf_values = np.cumsum(self.tilted_pdf_values)
        self.tilted_cdf_values /= self.tilted_cdf_values[-1]  # Ensure the last value is 1

    def get_execution_time(self):
        """
        Sample execution time using the original CDF (inverse transform sampling).
        """
        random_value = np.random.uniform(0, 1)
        idx = bisect_left(self.cdf_values, random_value)
        return self.x_values[idx]

    def get_log_pWCET_mgf(self, s):
        """
        Compute the log of the moment-generating function (MGF) for pWCET analysis.
        """
        upper_bound = self.abnormal_mu + 5 * self.abnormal_sigma  # Integration range
        a = s * upper_bound  # Log-Sum-Exp trick for numerical stability

        def log_integrand(x):
            # Weighted sum of normal and abnormal distributions
            pdf_value = (
                (1 - ABNORMAL_MODE_PROB) * self.normal_dist.pdf(x) +
                ABNORMAL_MODE_PROB * self.abnormal_dist.pdf(x)
            )
            return (s * x - a) + np.log(pdf_value)

        # Integrate and apply log-sum-exp correction
        log_pWCET_mgf, _ = quad(lambda x: np.exp(log_integrand(x)), 0, upper_bound)
        return a + np.log(log_pWCET_mgf)
