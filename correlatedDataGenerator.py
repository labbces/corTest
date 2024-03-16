#Generate large matrix with correlated data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy.stats import norm
from scipy.stats import multivariate_normal
# Set random seed
np.random.seed(0)
# Number of variables
num_vars = 30
# Number of data points
num_data = 100
# Generate random data
data = np.random.rand(num_data, num_vars)
print(data)

# Generate correlation matrix
corr = np.corrcoef(data, rowvar=False)

# Generate correlated data
data_corr = multivariate_normal.rvs(mean=np.zeros(num_vars), cov=corr, size=num_data)
# Plot data
sns.pairplot(pd.DataFrame(data_corr))
plt.show()