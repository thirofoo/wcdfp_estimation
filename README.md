# wcdfp_estimation

## Directory Structure

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

## Running Scripts

This project provides various scripts for evaluation, plotting, and verification. Development was conducted using `rye`, making it the recommended approach for running scripts. However, you can also use the `pip`-based alternative if preferred. Both options execute the same scripts, so choose the one that best fits your workflow.

---

### Option 1: Using rye (Recommended)

1. **Ensure Python 3.8 or later is installed.**

2. **Install rye:**
    ```bash
    curl -sSL https://install.rye-up.com | bash
    ```

3. **Verify the installation:**
    ```bash
    rye --version
    ```

4. **Synchronize your environment:**
    ```bash
    rye sync
    ```

**Run scripts with the `rye run <command>` prefix:**

#### Evaluation Scripts
```bash
rye run evaluate_monte_carlo                   # Monte Carlo evaluation.
rye run evaluate_monte_carlo_adjust_sample     # Adjust sample size in Monte Carlo evaluation.
rye run evaluate_berry_essen                   # Berry-Esseen theorem evaluation.
rye run evaluate_sequential_conv               # Circular Convolution following arrival order.
rye run evaluate_aggregate_conv_orig           # Circular Convolution using repeated squaring.
rye run evaluate_aggregate_conv_imp            # Circular Convolution with repeated squaring and optimized folding order.
rye run evaluate_one_taskset_all               # Plot normalized response times for one taskset.
rye run evaluate_all                           # Run all evaluations.
```

#### Plotting Scripts
```bash
rye run plot_wcdfp_comparison                  # Generate WCDFP comparison plots.
rye run plot_time_ratio_vs_wcdfp_ratio         # Plot time ratio versus WCDFP ratio.
rye run plot_execution_time                    # Create execution time boxplots.
rye run plot_comparison_for_task_id            # Compare results for a specific task.
```

#### Verification Scripts
```bash
rye run verify_taskset                         # Verify generated tasksets.
rye run verify_monte_carlo                     # Verify Monte Carlo results.
rye run verify_convolution                     # Verify Circular Convolution results.
rye run verify_berry_essen                     # Verify Berry-Esseen theorem results.
```

---

### Option 2: Using pip (Alternative)

1. **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    ```

2. **Activate the virtual environment:**
    ```bash
    # For Linux/macOS:
    source venv/bin/activate

    # For Windows:
    venv\Scripts\activate
    ```

3. **Install the project package:**
    ```bash
    pip3 install .
    ```

**After installation, run the scripts directly (without the rye prefix):**

#### Evaluation Scripts
```bash
evaluate_monte_carlo
evaluate_monte_carlo_adjust_sample
evaluate_berry_essen
evaluate_sequential_conv
evaluate_aggregate_conv_orig
evaluate_aggregate_conv_imp
evaluate_one_taskset_all
evaluate_all
```

#### Plotting Scripts
```bash
plot_wcdfp_comparison
plot_time_ratio_vs_wcdfp_ratio
plot_execution_time
plot_comparison_for_task_id
```

#### Verification Scripts
```bash
verify_taskset
verify_monte_carlo
verify_convolution
verify_berry_essen
```

---

### Additional Note on Plotting

Some plotting scripts support an optional `--mode` flag to change the coloring scheme:

- `--mode 0`: Default (no gradient)
- `--mode 1`: Gradient based on utilization rates
- `--mode 2`: Gradient based on task counts

Example usage:
```bash
rye run plot_time_ratio_vs_wcdfp_ratio -- --mode 2
```
or
```bash
plot_time_ratio_vs_wcdfp_ratio --mode 2
```

*Remember to sync or reinstall dependencies whenever you update project configurations.*

> [!NOTE]
> Running any evaluate script will generate evaluation results as CSV files saved in `src/evaluation/output/`. These CSV files serve as data sources for the subsequent plotting scripts. Ensure that the evaluation step is completed before running any plot commands.

## License

This project is licensed under the MIT License.
