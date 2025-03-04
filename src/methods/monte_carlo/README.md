# Monte Carlo Simulation

## Overview

This module provides tools for simulating and analyzing response times in real-time systems using Monte Carlo methods. It includes utilities for sample size estimation, response time simulation, and deadline miss probability calculations using both Agresti-Coull and Jeffreys intervals. Additionally, it offers support for parallel processing and visualization of response time distributions.

## Features

- **Sample Size Calculation:**  
  Compute the required sample size for binomial estimation based on a specified error margin and false probability.

- **Monte Carlo Simulation:**  
  Simulate the response time of a target job using a timeline-based approach to capture scheduling dynamics.

- **Deadline Miss Probability Estimation:**  
  Estimate deadline miss probabilities with statistical intervals (Agresti-Coull and Jeffreys) for robust analysis.  
  *Note: By default, the Agresti-Coull interval is used.*

- **Parallel Processing:**  
  Utilize multiple processes via `ProcessPoolExecutor` to speed up Monte Carlo simulations.
