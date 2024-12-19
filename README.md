# graduation research

## directory structure

```
project-root/
├── src/
│   ├── common/
│   │   ├── __init__.py
│   │   ├── job.py
│   │   ├── parameters.py
│   │   ├── task.py
│   │   ├── taskset.py
│   │   └── utils.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   └── analize.py
│   ├── methods/
│   │   ├── monte_carlo/
│   │   │   ├── __init__.py
│   │   │   └── estimation.py
│   │   ├── berry_essen/
│   │   │   ├── __init__.py
│   │   │   └── estimation.py
│   │   └── circular_convolution/
│   │       ├── __init__.py
│   │       └── estimation.py
│   └── verification/
│       ├── __init__.py
│       ├── verify_taskset.py
│       ├── verify_monte_carlo.py
│       ├── verify_berry_essen.py
│       └── verify_circular_convolution.py
├── README.md
├── pyproject.toml
└── requirements.lock
```

## How to Run the Evaluation

The project uses `rye` for script execution. Below are the available commands categorized into evaluation, plotting, and verification scripts. Use the `rye run <command>` format to execute these scripts. For instance, optional arguments like `--mode` can be appended for specific configurations (e.g., `rye run evaluate_monte_carlo -- --mode 1`).

## How to Run the Evaluation

The project uses `rye` for script execution. Below are the categorized commands available in the `pyproject.toml` configuration.

```bash
# Evaluation Scripts
rye run evaluate_monte_carlo                   # Run Monte Carlo evaluation.
rye run evaluate_monte_carlo_adjust_sample     # Adjust sample size in Monte Carlo evaluation.
rye run evaluate_berry_essen                   # Evaluate using the Berry-Esseen theorem.
rye run evaluate_convolution                   # Evaluate using Circular Convolution.
rye run evaluate_convolution_doubling          # Evaluate using Circular Convolution with Doubling.
rye run evaluate_all                           # Run evaluations for all methods.
rye run evaluate_one_taskset_all               # Plot normalized response times for one taskset across all methods.

# Plotting Scripts
rye run plot_wcdfp_comparison                  # Generate WCDFP comparison plots.
rye run plot_time_ratio_vs_wcdfp_ratio         # Plot time ratio versus WCDFP ratio.
rye run plot_execution_time                    # Create execution time boxplots.
rye run plot_comparison_for_task_id            # Compare results for a specific task ID.

# Verification Scripts
rye run verify_taskset                         # Verify the validity of the generated tasksets.
rye run verify_monte_carlo                     # Verify Monte Carlo results.
rye run verify_convolution                     # Verify Circular Convolution results.
rye run verify_berry_essen                     # Verify Berry-Esseen theorem results.
```

## Additional Notes

Some scripts support an optional `--mode` flag to customize the script behavior. This flag is specifically applicable to the following plotting scripts:

- **plot_time_ratio_vs_wcdfp_ratio**
- **plot_wcdfp_comparison**

The `--mode` flag modifies the coloring scheme of the plots:
- `--mode 0`: No gradient coloring (default).
- `--mode 1`: Gradient coloring based on utilization rates.
- `--mode 2`: Gradient coloring based on task counts.

For example:
```bash
rye run plot_time_ratio_vs_wcdfp_ratio -- --mode 1
```
