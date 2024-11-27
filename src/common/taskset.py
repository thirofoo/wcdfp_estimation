import random
import numpy as np
from common.task import Task
from common.job import Job
from common.utils import round_min_unit, calculate_wcet
from common.parameters import MINIMUM_TIME_UNIT


class TaskSet:
    """
    Represents a collection of tasks and provides methods to manage and generate jobs.
    """
    def __init__(self, task_num, utilization_rate, seed=0):
        self.task_num = task_num
        self.utilization_rate = utilization_rate
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Initialize task list, timeline, and arrival times
        self.tasks = []
        self.timeline = []
        self.arrival_times = []

        # Generate the task set
        self.generate_task_set()

        # Extract arrival times from the timeline
        for t, jobs_at_t in enumerate(self.timeline):
            if jobs_at_t:
                # Sort jobs at each time step by priority (RMS)
                jobs_at_t.sort(reverse=True)
                self.arrival_times.append(t)

    def generate_task_set(self):
        """
        Generate a task set based on the number of tasks and utilization rate.
        """
        dirichlet_dist = np.random.dirichlet(np.ones(self.task_num))
        execution_rates = dirichlet_dist * self.utilization_rate

        task_influences = []
        max_deadline = 0

        # Calculate task parameters
        for rate in execution_rates:
            minimum_inter_arrival_time = round_min_unit(
                np.exp(np.random.uniform(np.log(10), np.log(1000)))
            )
            relative_deadline = minimum_inter_arrival_time
            max_deadline = max(max_deadline, relative_deadline)

            wcet = calculate_wcet(minimum_inter_arrival_time, rate)
            influence = minimum_inter_arrival_time * rate
            task_influences.append((wcet, relative_deadline, minimum_inter_arrival_time, rate, influence))

        # Sort by influence and assign theta
        task_influences.sort(key=lambda x: x[-1], reverse=True)
        for i, (wcet, relative_deadline, minimum_inter_arrival_time, rate, influence) in enumerate(task_influences):
            theta = 1.0 if i < self.task_num * 0.3 else 0.0
            task = Task(wcet, relative_deadline, minimum_inter_arrival_time, theta)
            self.tasks.append(task)

        # Sort tasks by RMS priority (shorter period = higher priority)
        self.tasks.sort(key=lambda x: x.minimum_inter_arrival_time)

        # Create the timeline
        timeline_size = int((max_deadline + 1) / MINIMUM_TIME_UNIT)
        self.timeline = [[] for _ in range(timeline_size)]

        for task in self.tasks:
            t = 0
            while t <= max_deadline:
                job = Job(task, t)
                timeline_index = int(t / MINIMUM_TIME_UNIT)
                self.timeline[timeline_index].append(job)
                t += task.minimum_inter_arrival_time
