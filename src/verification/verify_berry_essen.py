import os
import matplotlib.pyplot as plt
from common.taskset import TaskSet
from methods.berry_essen.estimation import calculate_wcdfp_by_berry_essen


def verify_berry_essen():
    """
    Verify response time calculation using Berry-Essen theorem for the lowest priority task in a TaskSet.
    """
    # Create TaskSet
    seed = 3
    task_num = 100
    utilization_rate = 0.60
    taskset = TaskSet(task_num, utilization_rate, seed=seed)

    # Output directory for results
    output_dir = "src/verification/output"
    os.makedirs(output_dir, exist_ok=True)

    # Log details about the lowest priority task's job
    target_job = taskset.target_job
    print(f"lowest priority absolute_deadline : {target_job.absolute_deadline}")
    print(f"lowest priority execution_time : {target_job.task.get_execution_time()}\n")

    # Calculate response time using Berry-Essen (upper bound)
    print("Calculating response time using Berry-Essen (Upper)...")
    wcdfp_upper, (x_values_upper, berry_essen_cdf_upper) = calculate_wcdfp_by_berry_essen(
        taskset, target_job, log_flag=True, upper=True
    )
    print(f"WCDFP (Berry-Essen Upper): {wcdfp_upper}\n")

    # Calculate response time using Berry-Essen (lower bound)
    print("Calculating response time using Berry-Essen (Lower)...")
    wcdfp_lower, (x_values_lower, berry_essen_cdf_lower) = calculate_wcdfp_by_berry_essen(
        taskset, target_job, log_flag=True, upper=False
    )
    print(f"WCDFP (Berry-Essen Lower): {wcdfp_lower}")

    # Plot CDFs
    plt.figure(figsize=(10, 6))
    plt.plot(
        x_values_upper,
        berry_essen_cdf_upper,
        label="Berry-Essen CDF (Upper)",
        color="blue",
        drawstyle="steps-mid",
    )
    plt.plot(
        x_values_lower,
        berry_essen_cdf_lower,
        label="Berry-Essen CDF (Lower)",
        color="green",
        linestyle="--",
        drawstyle="steps-mid",
    )

    # Deadline line
    deadline = target_job.absolute_deadline
    plt.axvline(x=deadline, color="red", linestyle="--", label="Deadline")

    # Plot settings
    plt.xlabel("Response Time")
    plt.ylabel("Cumulative Distribution Function")
    plt.title("Berry-Essen CDF for Target Job")
    plt.legend()
    plt.grid(True)

    # Save plot to output directory
    plot_path = os.path.join(output_dir, "berry_essen_cdf_bounds.png")
    plt.savefig(plot_path)
    print(f"Berry-Essen CDF plot (with bounds) saved to {plot_path}")
    plt.close()
