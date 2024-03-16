import argparse
import pandas as pd
import deepgraph as dg
from memory_profiler import memory_usage
import multiprocessing
from time import time

def compute_correlations(file_name, n_cores):
    # Load data from file
    v = pd.read_csv(file_name)
    g = dg.DeepGraph(v)

    # Start timer
    start_time = time()

    # Compute the Pearson correlation between pairs of rows in parallel
    gt = g.create_edges(connectors=[dg.corr], no_duplicates=True, parallel=n_cores)

    # Stop timer
    elapsed_time = time() - start_time
    return elapsed_time

def main():
    parser = argparse.ArgumentParser(description='Compute Pairwise Correlations with DeepGraph.')
    parser.add_argument('file_name', type=str, help='The name of the file containing the dataset.')
    args = parser.parse_args()

    # Test the performance for different numbers of cores, starting with the largest number
    max_cores = multiprocessing.cpu_count()
    results = []

    # Note the change here: iterating in reverse order
    for n_cores in range(max_cores, 0, -1):
        peak_memory = max(memory_usage((compute_correlations, (args.file_name, n_cores))))
        elapsed_time = compute_correlations(args.file_name, n_cores)
        results.append((n_cores, peak_memory, elapsed_time))
        print(f"Cores: {n_cores}, Peak Memory: {peak_memory:.2f} MiB, Time: {elapsed_time:.2f} s")

if __name__ == "__main__":
    main()
