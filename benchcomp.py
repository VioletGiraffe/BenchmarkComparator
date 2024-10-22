import subprocess
import time
import numpy as np
from scipy import stats
import math

# Function to run a version and return the execution time
def run_version(command):
    start = time.time()
    subprocess.run(command, shell=True)
    end = time.time()
    return end - start

# Run the benchmark for each version and collect results
def benchmark(version_a_cmd, version_b_cmd, runs):
    times_a = [run_version(version_a_cmd) for _ in range(runs)]
    times_b = [run_version(version_b_cmd) for _ in range(runs)]
    return np.array(times_a), np.array(times_b)

# Calculate the number of runs needed based on desired confidence and power
def calculate_sample_size(std_a, std_b, mean_diff, alpha=0.05, power=0.8):
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    
    pooled_std = math.sqrt((std_a ** 2 + std_b ** 2) / 2)
    effect_size = mean_diff / pooled_std
    sample_size = (2 * (z_alpha + z_beta) ** 2 * (pooled_std ** 2)) / mean_diff ** 2
    return math.ceil(sample_size)

# Perform a t-test to check if the difference between the two versions is significant
def perform_ttest(times_a, times_b):
    t_stat, p_value = stats.ttest_ind(times_a, times_b)
    return t_stat, p_value

# Example usage
if __name__ == "__main__":
    version_a_cmd = "./version_A"
    version_b_cmd = "./version_B"
    runs = 30  # Start with 30 runs per version

    times_a, times_b = benchmark(version_a_cmd, version_b_cmd, runs)
    
    # Calculate statistics
    mean_a, std_a = np.mean(times_a), np.std(times_a, ddof=1)
    mean_b, std_b = np.mean(times_b), np.std(times_b, ddof=1)
    
    # Perform the t-test
    t_stat, p_value = perform_ttest(times_a, times_b)
    print(f"T-statistic: {t_stat}, P-value: {p_value}")

    if p_value < 0.05:
        print("The difference is statistically significant!")
    else:
        print("No significant difference detected.")

    # Calculate the minimum number of runs needed
    min_sample_size = calculate_sample_size(std_a, std_b, abs(mean_a - mean_b))
    print(f"Minimum number of runs needed: {min_sample_size}")
