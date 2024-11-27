import os
import time
import numpy as np
import matplotlib.pyplot as plt
from methods.monte_carlo.estimation import calculate_response_time_distribution, required_sample_size_binomial
from common.taskset import TaskSet

def main():
    """
    Main function to evaluate Monte Carlo response time distributions for multiple task sets.
    """
    total_taskset = 1

    # Monte Carlo parameters
    error_margin = 0.01
    task_num = 100
    utilization_rate = 0.65
    thread_num = 16
    seed = 3
    false_probability = 0.000001

    # Calculate required sample size
    samples = required_sample_size_binomial(error_margin, false_probability)
    print(f"Required Sample Size: {samples}")

    # Output directory for results
    output_dir = "src/evaluation/output"
    os.makedirs(output_dir, exist_ok=True)

    # Evaluate each task set
    for i in range(total_taskset):
        print(f"\nEvaluating TaskSet {i + 1}/{total_taskset}...")

        # Generate TaskSet with different seed for each iteration
        current_seed = seed + i
        taskset = TaskSet(task_num=task_num, utilization_rate=utilization_rate, seed=current_seed)
        print(f"Generated TaskSet {i + 1}.")

        # Calculate Monte Carlo distribution
        start_time = time.time()
        response_times_mc, wcdfp_mc = calculate_response_time_distribution(
            taskset=taskset,
            target_job=taskset.timeline[0][-1],
            false_probability=false_probability,
            thread_num=thread_num,
            log_flag=True,
            plot_flag=True,
            samples=samples,
        )
        sorted_times_mc = np.sort(response_times_mc)
        monte_carlo_cdf_values = np.arange(1, len(sorted_times_mc) + 1) / len(sorted_times_mc)

        print(f"Monte Carlo WCDFP: {wcdfp_mc}")
        print(f"Elapsed Time: {time.time() - start_time:.2f} sec")

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.step(
            sorted_times_mc,
            monte_carlo_cdf_values,
            label="Monte Carlo CDF",
            where="post",
        )

        # Deadline line
        deadline = taskset.timeline[0][-1].absolute_deadline
        plt.axvline(x=deadline, color="red", linestyle="--", label="Deadline")

        # Plot settings
        plt.xlabel("Execution Time")
        plt.ylabel("Cumulative Distribution Function")
        plt.title(f"Monte Carlo CDF for TaskSet {i + 1}")
        plt.legend()
        plt.grid(True)

        # Save the plot to the output directory
        plot_path = os.path.join(output_dir, f"monte_carlo_cdf_{i + 1}.png")
        plt.savefig(plot_path)
        print(f"Monte Carlo CDF plot saved to {plot_path}")
        plt.close()

    print("\nMonte Carlo evaluation completed for all task sets.")

if __name__ == "__main__":
    main()
