# Minimum time unit (1 = 1ms)
MINIMUM_TIME_UNIT = 0.001

# Probability of abnormal execution mode
ABNORMAL_MODE_PROB = 0.05

# Constants for execution time distribution
NORMAL_MEAN_COEFF = 3.0    # Mean for normal execution mode
NORMAL_STD_COEFF = 10.0    # Standard deviation for normal mode
ABNORMAL_MEAN_COEFF = 1.2  # Mean for abnormal execution mode
ABNORMAL_STD_COEFF = 30.0  # Standard deviation for abnormal mode

# Threshold to determine if the execution time distribution is sparse
SPARSITY_THRESHOLD = 30

# Coefficient used in the Berry-Essen theorem to adjust the penalty term (ψ).
# This value (0.56) is based on the latest research findings as the optimal parameter for accuracy.
BERRY_ESSEN_COEFFICIENT = 0.56
