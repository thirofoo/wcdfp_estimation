import numpy as np
from scipy.signal import fftconvolve
from tqdm import tqdm
from common.parameters import MINIMUM_TIME_UNIT


def convolve_and_truncate(pdf1, pdf2, size):
    """
    Perform convolution of two probability density functions and truncate the result.

    :param pdf1: First PDF
    :param pdf2: Second PDF
    :param size: Maximum size of the result (truncation point)
    :return: Truncated PDF and exceed probability
    """
    convolved_pdf = fftconvolve(pdf1, pdf2, mode='full')
    truncated_pdf = np.maximum(convolved_pdf[:size], 0.0)
    exceed_prob = np.sum(convolved_pdf[size:])
    return truncated_pdf, exceed_prob


def convolve(pdf1, pdf2):
    convolved_pdf = fftconvolve(pdf1, pdf2, mode='full')
    truncated_pdf = np.maximum(convolved_pdf, 0.0)
    return truncated_pdf


def calculate_response_time_by_conv(taskset, target_job, log_flag=False, float128_flag=False):
    """
    Calculate response time distribution using convolution with truncation.

    :param taskset: TaskSet containing tasks and timeline
    :param target_job: Target job for which response time distribution is calculated
    :param log_flag: If True, logs detailed intermediate results
    :param traditional_ci: If True, ignores carry-in at time 0
    :param float128_flag: If True, use float128 precision; otherwise, use float64 precision
    :return: Response time distribution and WCDFP
    """
    dtype = np.float128 if float128_flag else np.float64

    size = int(target_job.absolute_deadline / MINIMUM_TIME_UNIT) + 1
    response_time = np.zeros(size, dtype=dtype)
    response_time[0] = dtype(1.0)  # Initial PDF for response time
    true_response_time = np.zeros(size, dtype=dtype)  # Track completed response times
    wcdfp = dtype(0.0)  # Initialize WCDFP as 0.0
    arrival_times = taskset.arrival_times
    timeline = taskset.timeline

    # Carry-in calculation
    for job in timeline[0]:
        if job < target_job:
            break
        normalized_pdf = job.task.original_pdf_values.astype(dtype)
        normalized_pdf /= np.sum(normalized_pdf)  # Normalize to sum to 1
        response_time, exceed_prob = convolve_and_truncate(response_time, normalized_pdf, size)
        wcdfp += exceed_prob

    for idx in tqdm(range(len(arrival_times)), desc="Processing arrival times", disable=not log_flag):
        t = arrival_times[idx] * MINIMUM_TIME_UNIT

        # Transfer completed response times to `true_response_time`
        for i in range(0 if idx == 0 else arrival_times[idx - 1], arrival_times[idx]):
            true_response_time[i] += response_time[i]
            response_time[i] = dtype(0.0)

        # Stop processing if we reach or exceed the deadline
        if t >= target_job.absolute_deadline:
            break

        # Process jobs at the current time
        for job in timeline[arrival_times[idx]]:
            if job < target_job or job == target_job:
                continue
            normalized_pdf = job.task.original_pdf_values.astype(dtype)
            normalized_pdf /= np.sum(normalized_pdf)  # Normalize to sum to 1
            response_time, exceed_prob = convolve_and_truncate(response_time, normalized_pdf, size)
            wcdfp += exceed_prob

    # Final transfer of accumulated `true_response_time` back to `response_time`
    for i in range(len(true_response_time)):
        response_time[i] += true_response_time[i]

    if log_flag:
        print(f"Final WCDFP: {wcdfp}")
        print(f"Sum of response_time: {np.sum(response_time)}")

    return response_time, wcdfp


def calculate_response_time_with_doubling(taskset, target_job, log_flag=False, float128_flag=False):
    """
    Calculate response time distribution using doubling technique with truncation.

    :param taskset: TaskSet containing tasks
    :param target_job: Target job for which response time distribution is calculated
    :param log_flag: If True, logs detailed intermediate results
    :param float128_flag: If True, use float128 precision; otherwise, use float64 precision
    :return: Response time distribution and WCDFP
    """
    dtype = np.float128 if float128_flag else np.float64

    size = int(target_job.absolute_deadline / MINIMUM_TIME_UNIT) + 1
    response_time = np.array([dtype(1.0)], dtype=dtype)  # Initial PDF for response time
    wcdfp = dtype(0.0)

    for task in tqdm(taskset.tasks, desc="Processing tasks", disable=not log_flag):
        release_count = int(np.ceil((target_job.task.relative_deadline + task.relative_deadline) / task.minimum_inter_arrival_time))
        if target_job.task == task:
            release_count = 1

        current_pdf = task.original_pdf_values.astype(dtype)
        current_pdf /= np.sum(current_pdf)
        while release_count > 0:
            if release_count % 2 == 1:
                response_time, exceed_prob = convolve_and_truncate(response_time, current_pdf, size)
                wcdfp += exceed_prob

            current_pdf, exceed_prob_doubling = convolve_and_truncate(current_pdf, current_pdf, size)
            current_pdf = np.concatenate((current_pdf, [exceed_prob_doubling])).astype(dtype)
            current_pdf /= np.sum(current_pdf)
            release_count //= 2

    if log_flag:
        print(f"Final WCDFP: {wcdfp}")
        print(f"Sum of response_time: {np.sum(response_time)}")

    return response_time, wcdfp
