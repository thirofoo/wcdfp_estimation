# Berry-Essen Method

## Overview

This module provides a method for calculating the Worst-Case Deadline Failure Probability (WCDFP) in real-time systems using the Berry–Esseen theorem. It also computes expected task statistics considering both normal and abnormal modes.

## Features

- **Expected Stats Calculation:**  
  Computes the expected mean and variance for a task by weighing its normal and abnormal mode parameters.

- **WCDFP Calculation via Berry–Esseen:**  
  Aggregates statistics across tasks and adjusts the cumulative distribution function (CDF) with a penalty term based on execution time samples to estimate WCDFP.
