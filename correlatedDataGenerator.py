import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_correlated_data_with_variable_noise(mean, cov, rows, noise_mean, noise_std):
    base_data = np.random.multivariate_normal(mean, cov, rows)
    noise_level = np.abs(np.random.normal(noise_mean, noise_std))
    noise = np.random.normal(0, noise_level, base_data.shape)
    noisy_data = base_data + noise
    return noisy_data

def create_patterns(num_patterns, num_columns):
    patterns = []
    for _ in range(num_patterns):
        mean = np.random.rand(num_columns) * 10
        cov = np.random.rand(num_columns, num_columns)
        cov = np.dot(cov, cov.transpose())
        patterns.append((mean, cov))
    return patterns

def write_data_to_file(filename, num_patterns, num_columns, noise_mean, noise_std, min_reps, max_reps, max_rows):
    total_rows = 0
    with open(filename, 'w') as f:
        for mean, cov in create_patterns(num_patterns, num_columns):
            if total_rows >= max_rows:
                break  # Stop if we have reached or exceeded the max_rows limit
            pattern_repeats = np.random.randint(min_reps, max_reps + 1)
            # Adjust pattern_repeats if adding them would exceed max_rows
            if total_rows + pattern_repeats > max_rows:
                pattern_repeats = max_rows - total_rows
            data = generate_correlated_data_with_variable_noise(mean, cov, pattern_repeats, noise_mean, noise_std)
            for row in data:
                f.write(','.join(map(str, row)) + '\n')
            total_rows += pattern_repeats

def read_data_from_file(filename):
    return np.loadtxt(filename, delimiter=',')

def plot_clustered_heatmap(data):
    correlation_matrix = np.corrcoef(data, rowvar=True)
    sns.clustermap(correlation_matrix, metric="correlation", standard_scale=1, cmap='coolwarm', figsize=(13, 10))
    plt.show()

def main():
    #Number of clusters present in the data
    num_patterns = 100
    #number of conditions/columns
    num_columns = 30
    filename = 'correlated_matrix.txt'
    noise_mean = 0.2
    noise_std = 0.8
    min_reps = 2
    max_reps = 100
    max_rows = 1000  # Approximate maximum number of rows

    write_data_to_file(filename, num_patterns, num_columns, noise_mean, noise_std, min_reps, max_reps, max_rows)

    data = read_data_from_file(filename)
    print(f"Dimensions of the data matrix: {data.shape}")

    plot_clustered_heatmap(data)

if __name__ == "__main__":
    main()
