import numpy as np
from scipy.signal import fftconvolve
from tqdm import tqdm
import heapq
from common.parameters import MINIMUM_TIME_UNIT
from typing import Tuple, List

def normalize_pdf(pdf: np.ndarray, dtype: type) -> np.ndarray:
    normalized = pdf.astype(dtype)
    total = np.sum(normalized)
    return normalized / total if total != 0 else normalized

def convolve_and_truncate(pdf1: np.ndarray, pdf2: np.ndarray, size: int) -> Tuple[np.ndarray, np.float64]:
    """
    Convolve two PDFs and truncate the result at the given size.
    Returns the truncated PDF and the probability mass that exceeds the size.
    """
    conv_pdf = fftconvolve(pdf1, pdf2, mode='full')
    truncated = np.maximum(conv_pdf[:size], 0.0)
    exceed_prob = np.sum(conv_pdf[size:])
    return truncated, exceed_prob

def convolve(pdf1: np.ndarray, pdf2: np.ndarray) -> np.ndarray:
    conv_pdf = fftconvolve(pdf1, pdf2, mode='full')
    return np.maximum(conv_pdf, 0.0)

def calculate_wcdfp_by_sequential_conv(taskset, target_job, log_flag: bool = False, float128_flag: bool = False) -> Tuple[np.ndarray, np.float64]:
    """
    Calculate response time distribution via convolution with truncation.
    """
    dtype = np.float128 if float128_flag else np.float64
    size = int(target_job.absolute_deadline / MINIMUM_TIME_UNIT) + 1

    # Initialize PDFs
    response_time = np.zeros(size, dtype=dtype)
    response_time[0] = dtype(1.0)
    true_response_time = np.zeros(size, dtype=dtype)
    wcdfp = dtype(0.0)

    arrival_times = taskset.arrival_times
    timeline = taskset.timeline

    # Process carry-in jobs using timeline[0]
    for job in timeline[0]:
        if job < target_job:
            break
        norm_pdf = normalize_pdf(job.task.original_pdf_values, dtype)
        response_time, exceed_prob = convolve_and_truncate(response_time, norm_pdf, size)
        wcdfp += exceed_prob

    # Process jobs at each arrival time
    for idx in tqdm(range(len(arrival_times)), desc="Processing arrival times", disable=not log_flag):
        current_time = arrival_times[idx] * MINIMUM_TIME_UNIT
        # Transfer completed response times
        start_idx = 0 if idx == 0 else arrival_times[idx - 1]
        for i in range(start_idx, arrival_times[idx]):
            true_response_time[i] += response_time[i]
            response_time[i] = dtype(0.0)
        if current_time >= target_job.absolute_deadline:
            break

        for job in timeline[arrival_times[idx]]:
            # Skip jobs that are equal to or come before the target job
            if job < target_job or job == target_job:
                continue
            norm_pdf = normalize_pdf(job.task.original_pdf_values, dtype)
            response_time, exceed_prob = convolve_and_truncate(response_time, norm_pdf, size)
            wcdfp += exceed_prob

    # Merge accumulated true response times back into response_time
    response_time += true_response_time

    if log_flag:
        print(f"Final WCDFP: {wcdfp}")
        print(f"Sum of response_time: {np.sum(response_time)}")

    return response_time, wcdfp

def calculate_wcdfp_by_aggregate_conv_orig(taskset, target_job, log_flag: bool = False, float128_flag: bool = False) -> Tuple[np.ndarray, np.float64]:
    """
    Calculate the response time distribution using the exponentiation by squaring technique.
    """
    dtype = np.float128 if float128_flag else np.float64
    size = int(target_job.absolute_deadline / MINIMUM_TIME_UNIT) + 1

    # Start with a PDF that is just a point mass at 0.
    response_time = np.array([dtype(1.0)], dtype=dtype)
    wcdfp = dtype(0.0)

    for task in tqdm(taskset.tasks, desc="Processing tasks", disable=not log_flag):
        # Determine how many releases to consider.
        release_count = int(np.ceil((target_job.task.relative_deadline + task.relative_deadline) / task.minimum_inter_arrival_time))
        if task == target_job.task:
            release_count = 1

        current_pdf = normalize_pdf(task.original_pdf_values, dtype)
        while release_count > 0:
            if release_count % 2 == 1:
                response_time, exceed_prob = convolve_and_truncate(response_time, current_pdf, size)
                wcdfp += exceed_prob
            # Double the current_pdf
            current_pdf, exceed_prob_dbl = convolve_and_truncate(current_pdf, current_pdf, size)
            # Append the exceed probability at the end as an extra element.
            current_pdf = np.concatenate((current_pdf, [exceed_prob_dbl])).astype(dtype)
            current_pdf = normalize_pdf(current_pdf, dtype)
            release_count //= 2

    if log_flag:
        print(f"Final WCDFP: {wcdfp}")
        print(f"Sum of response_time: {np.sum(response_time)}")

    return response_time, wcdfp

def calculate_wcdfp_by_aggregate_conv_imp(taskset, target_job, log_flag: bool = False, float128_flag: bool = False) -> Tuple[np.ndarray, np.float64]:
    """
    Calculate the response time distribution using exponentiation by squaring with a priority queue.
    """
    dtype = np.float128 if float128_flag else np.float64
    size = int(target_job.absolute_deadline / MINIMUM_TIME_UNIT) + 1

    response_pdfs: List[np.ndarray] = []
    wcdfp = dtype(0.0)
    pq = []  # Priority queue holding tuples of (pdf length, index)

    for task in tqdm(taskset.tasks, desc="Processing tasks", disable=not log_flag):
        release_count = int(np.ceil((target_job.task.relative_deadline + task.relative_deadline) / task.minimum_inter_arrival_time))
        if task == target_job.task:
            release_count = 1

        current_pdf = normalize_pdf(task.original_pdf_values, dtype)
        while release_count > 0:
            if release_count % 2 == 1:
                index = len(response_pdfs)
                response_pdfs.append(current_pdf)
                heapq.heappush(pq, (len(current_pdf), index))
            current_pdf, exceed_prob_dbl = convolve_and_truncate(current_pdf, current_pdf, size)
            current_pdf = np.concatenate((current_pdf, [exceed_prob_dbl])).astype(dtype)
            current_pdf = normalize_pdf(current_pdf, dtype)
            release_count //= 2

    # Merge all PDFs using a priority queue based on PDF length.
    while len(pq) > 1:
        _, idx1 = heapq.heappop(pq)
        _, idx2 = heapq.heappop(pq)
        merged, exceed_prob = convolve_and_truncate(response_pdfs[idx1], response_pdfs[idx2], size)
        response_pdfs[idx1] = merged
        wcdfp += exceed_prob
        heapq.heappush(pq, (len(merged), idx1))

    if log_flag:
        final_pdf_idx = pq[0][1]
        print(f"Final WCDFP: {wcdfp}")
        print(f"Sum of response_time: {np.sum(response_pdfs[final_pdf_idx])}")

    final_pdf = response_pdfs[pq[0][1]]
    return final_pdf, wcdfp
