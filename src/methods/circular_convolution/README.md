# Circular Convolution Methods

## Overview

This module provides implementations for computing response time distributions and the Worst-Case Deadline Failure Probability (WCDFP) in real-time systems using convolution-based methods. The module includes several algorithms, such as sequential convolution and two aggregate convolution approaches (one using exponentiation by squaring and another using a priority queue), to efficiently combine probability density functions (PDFs) with truncation.

## Features

- **Convolution with Truncation:**  
  Perform convolution of PDFs with truncation, returning both the truncated PDF and the probability mass that exceeds the specified size.

- **Sequential Convolution:**  
  Calculate WCDFP by sequentially convolving PDFs based on job arrival times and aggregating the results.

- **Aggregate Convolution (Exponentiation by Squaring):**  
  Use exponentiation by squaring to efficiently compute the overall response time distribution.

- **Aggregate Convolution with Priority Queue:**  
  Merge PDFs using a priority queue for improved efficiency in handling convolution results.
