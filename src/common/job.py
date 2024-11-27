class Job:
    """
    Represents a single job in the task scheduling system.
    """
    _id_counter = 0  # Class variable to maintain a unique ID counter

    def __init__(self, task, release_time):
        """
        Initialize a job with its associated task, release time, and deadline.
        """
        self.task = task  # Reference to the associated task
        self.release_time = release_time  # Release time of the job
        self.absolute_deadline = release_time + task.relative_deadline  # Compute absolute deadline
        self.id = self.get_id()  # Assign a unique ID

    @classmethod
    def get_id(cls):
        """
        Generate a unique ID for each job.
        """
        cls._id_counter += 1
        return cls._id_counter

    def __lt__(self, other):
        """
        Compare jobs for priority in Rate-Monotonic Scheduling (RMS).
        Jobs with shorter minimum inter-arrival times have higher priority.
        If two jobs have the same inter-arrival time, compare by absolute deadline.
        """
        # 1. Higher priority for shorter minimum inter-arrival times
        if self.task.minimum_inter_arrival_time != other.task.minimum_inter_arrival_time:
            return other.task.minimum_inter_arrival_time < self.task.minimum_inter_arrival_time
        # 2. If inter-arrival times are equal, compare by absolute deadline
        return other.absolute_deadline < self.absolute_deadline

    def __eq__(self, other):
        """
        Check if two jobs belong to the same task.
        """
        return self.task == other.task
