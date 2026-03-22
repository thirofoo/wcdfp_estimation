import heapq
from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import fftconvolve
from tqdm import tqdm

from common.parameters import MINIMUM_TIME_UNIT

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


def conservative_downsample_to_next_stage(pdf: np.ndarray, dtype: type) -> np.ndarray:
    """
    Downsample a PDF by a factor of 2 and conservatively move probability mass
    to the later (worse) time side.
    Pairing rule:
      (0,1) -> 1, (2,3) -> 2, ...
    """
    n = len(pdf)
    pair_count = n // 2
    out_len = (n + 1) // 2 + 1
    downsampled = np.zeros(out_len, dtype=dtype)

    if pair_count > 0:
        downsampled[1:1 + pair_count] += pdf[0:2 * pair_count:2] + pdf[1:2 * pair_count:2]
    if n % 2 == 1:
        downsampled[1 + pair_count] += pdf[-1]

    return np.maximum(downsampled, dtype(0.0))


def deadline_size_for_stage(absolute_deadline: float, stage: int) -> int:
    unit = MINIMUM_TIME_UNIT * (2 ** stage)
    return int(absolute_deadline / unit) + 1

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


def calculate_wcdfp_by_aggregate_conv_imp_rescaled(
    taskset,
    target_job,
    log_flag: bool = False,
    float128_flag: bool = False,
    l_max: int = 5000,
) -> Tuple[np.ndarray, np.float64]:
    """
    Aggregate convolution with multi-stage conservative rescaling.
    If PDF length exceeds l_max, the PDF is downsampled by 2 and promoted to
    the next stage (coarser resolution).
    """
    if l_max < 2:
        raise ValueError("l_max must be >= 2")

    dtype = np.float128 if float128_flag else np.float64
    abs_deadline = target_job.absolute_deadline
    stage0_size = deadline_size_for_stage(abs_deadline, 0)

    # No rescaling can occur in this case, so use the original implementation
    # to preserve baseline behavior exactly.
    if l_max >= stage0_size + 1:
        return calculate_wcdfp_by_aggregate_conv_imp(
            taskset=taskset,
            target_job=target_job,
            log_flag=log_flag,
            float128_flag=float128_flag,
        )

    stage_queues: Dict[int, List[Tuple[int, int, np.ndarray]]] = {}
    push_counter = 0
    total_components = 0
    wcdfp = dtype(0.0)

    def push(stage: int, pdf: np.ndarray) -> None:
        nonlocal push_counter
        if stage not in stage_queues:
            stage_queues[stage] = []
        push_counter += 1
        heapq.heappush(stage_queues[stage], (len(pdf), push_counter, pdf))

    def pop(stage: int) -> np.ndarray:
        _, __, pdf = heapq.heappop(stage_queues[stage])
        if not stage_queues[stage]:
            del stage_queues[stage]
        return pdf

    def compact_by_lmax(pdf: np.ndarray, stage: int) -> Tuple[np.ndarray, int]:
        while len(pdf) > l_max:
            pdf = conservative_downsample_to_next_stage(pdf, dtype)
            stage += 1
        return pdf, stage

    # Phase 1: per-task exponentiation by squaring with adaptive rescaling.
    for task in tqdm(taskset.tasks, desc="Processing tasks", disable=not log_flag):
        release_count = int(np.ceil((target_job.task.relative_deadline + task.relative_deadline) / task.minimum_inter_arrival_time))
        if task == target_job.task:
            release_count = 1

        current_pdf = normalize_pdf(task.original_pdf_values, dtype)
        current_stage = 0
        current_pdf, current_stage = compact_by_lmax(current_pdf, current_stage)

        while release_count > 0:
            if release_count % 2 == 1:
                push(current_stage, current_pdf)
                total_components += 1

            stage_size = deadline_size_for_stage(abs_deadline, current_stage)
            current_pdf, exceed_prob_dbl = convolve_and_truncate(current_pdf, current_pdf, stage_size)
            current_pdf = np.concatenate((current_pdf, [exceed_prob_dbl])).astype(dtype)
            current_pdf = normalize_pdf(current_pdf, dtype)
            current_pdf, current_stage = compact_by_lmax(current_pdf, current_stage)
            release_count //= 2

    # Phase 2: multi-stage Huffman-like merge.
    while total_components > 1:
        stage = min(stage_queues.keys())
        queue_len = len(stage_queues[stage])

        if queue_len >= 2:
            pdf1 = pop(stage)
            pdf2 = pop(stage)
            total_components -= 1  # Two components merged into one.

            stage_size = deadline_size_for_stage(abs_deadline, stage)
            merged, exceed_prob = convolve_and_truncate(pdf1, pdf2, stage_size)
            wcdfp += exceed_prob

            merged_stage = stage
            merged, merged_stage = compact_by_lmax(merged, merged_stage)
            push(merged_stage, merged)
            continue

        # Move lone PDF to next stage with conservative downsampling.
        lone = pop(stage)
        next_stage = stage + 1
        promoted = conservative_downsample_to_next_stage(lone, dtype)
        promoted, next_stage = compact_by_lmax(promoted, next_stage)
        push(next_stage, promoted)

    # Collect the final PDF.
    if not stage_queues:
        final_pdf = np.array([dtype(1.0)], dtype=dtype)
    else:
        final_stage = min(stage_queues.keys())
        final_pdf = pop(final_stage)

    if log_flag:
        print(f"Final WCDFP (rescaled): {wcdfp}")
        print(f"Remaining PDF length: {len(final_pdf)}")

    return final_pdf, wcdfp
