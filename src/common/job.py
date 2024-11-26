class Job:
    """
    Represents a single job in the task scheduling system.
    """
    _id_counter = 0  # ID counter for uniquely identifying jobs

    def __init__(self, task, release_time):
        self.task = task
        self.release_time = release_time
        self.absolute_deadline = release_time + task.relative_deadline
        self.id = self.get_id()

    @classmethod
    def get_id(cls):
        """
        Generate a unique ID for each job.
        """
        cls._id_counter += 1
        return cls._id_counter

    def __lt__(self, other):
        """
        Priority comparison for Rate-Monotonic Scheduling.
        Jobs with shorter minimum inter-arrival times have higher priority.
        """
        return self.task.minimum_inter_arrival_time < other.task.minimum_inter_arrival_time
