import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from common.taskset import Task

def test_task():
    """
    Test the Task class by generating a task and plotting its execution time distribution.
    """
    # Create the output directory
    output_dir = Path("src/tests/img")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a test Task
    test_task = Task(wcet=10, relative_deadline=10, minimum_inter_arrival_time=10, theta=2.0)

    # First Plot: CDF
    plt.figure(figsize=(10, 6))
    plt.step(test_task.x_values, test_task.cdf_values, label="CDF", where='post', color='gray')
    plt.xlabel("Execution Time")
    plt.ylabel("Cumulative Distribution Function")
    plt.title("Step CDF")
    plt.legend()
    plt.grid()
    plt.savefig(output_dir / "step_cdf.png")  # Save plot to file
    plt.close()

    # Second Plot: Original Sampling
    sample_num = 10000
    original_samples = [test_task.get_execution_time() for _ in range(sample_num)]

    bins = np.linspace(0, test_task.abnormal_mu + 3 * test_task.abnormal_sigma, 20)

    # Plot Original Sampling and PDF
    plt.figure(figsize=(10, 6))
    plt.hist(original_samples, bins=bins, edgecolor='black', alpha=0.6, density=True, label="Original Sampling", color='pink')
    plt.plot(test_task.x_values, test_task.original_pdf_values, label="Original PDF", color='red', linestyle='--', linewidth=2)
    plt.title("Original Sampling and PDF")
    plt.xlabel("Execution Time")
    plt.ylabel("Density")
    plt.legend()
    plt.grid()
    plt.savefig(output_dir / "original_sampling_pdf.png")  # Save plot to file
    plt.close()

if __name__ == "__main__":
    test_task()
