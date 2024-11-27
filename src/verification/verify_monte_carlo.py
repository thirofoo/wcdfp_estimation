from common.taskset import TaskSet
from methods.monte_carlo.estimation import calculate_response_time

def verify_monte_carlo():
    """
    Verify the response time calculation for the lowest priority task in a TaskSet.
    """
    # Create TaskSet
    seed = 3
    task_num = 100
    utilization_rate = 0.60
    taskset = TaskSet(task_num, utilization_rate, seed=seed)

    # Log details about the lowest priority task's job
    print(f"timeline[0][-1] absolute_deadline : {taskset.timeline[0][-1].absolute_deadline}")
    print(f"timeline[0][-1] execution_time : {taskset.timeline[0][-1].task.get_execution_time()}\n")

    # Calculate response time
    response_time = calculate_response_time(taskset, taskset.timeline[0][-1], log_flag=True)
    print(f"Calculated Response Time: {response_time}")

if __name__ == "__main__":
    verify_monte_carlo()
