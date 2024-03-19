#conda create -n corTest deepgraph numpy pandas tables matplot seaborn
import argparse
import shutil
import deepgraph as dg
import numpy as np
import pandas as pd
import os
#from multiprocessing import Pool
import multiprocessing as mp
from datetime import datetime

# Get the current date and time
current_time = datetime.now()

parser = argparse.ArgumentParser(description='Compute Pairwise Correlations with DeepGraph.')
parser.add_argument('file_name', type=str, help='The name of the file containing the dataset.')
parser.add_argument('--max_cores', type=int, default=mp.cpu_count(), help='Maximum number of cores to use.')
parser.add_argument('--step_size', type=int, default=1e5, help='chunk size.')
parser.add_argument('--n_rows', type=int, default=1e5, help='Numbre of rows.')
parser.add_argument('--n_cols', type=int, default=30, help='Number of columns.')

args = parser.parse_args()


# parameters (change these to control RAM usage)
n_processes = args.max_cores
n_rows = args.n_rows
n_cols = args.n_cols
step_size = args.step_size

TMPpath='tmp/correlations_'+current_time.strftime('%Y-%m-%d_%H-%M-%S')

# load samples as memory-map
X = np.memmap(args.file_name, mode='r', shape=(n_rows,n_cols))


print(f'shape0: {X.shape[0]}\n')
# create node table that stores references to the mem-mapped samples
v = pd.DataFrame({'index': range(X.shape[0])})

# connector function to compute pairwise pearson correlations
def corr(index_s, index_t):
#    print(f's: {index_s}\n')
#    print(f't: {index_t}\n')
    features_s = X[index_s]
    features_t = X[index_t]
    corr = np.einsum('ij,ij->i', features_s, features_t) / n_cols
    return corr

# index array for parallelization
pos_array = np.array(np.linspace(0, n_rows*(n_rows-1)//2, n_processes), dtype=int)

# parallel computation
def create_ei(i):

    print(f'i: {i}\n')
    from_pos = pos_array[i]
    to_pos = pos_array[i+1]

    # initiate DeepGraph
    g = dg.DeepGraph(v)

    # create edges
    g.create_edges(connectors=corr, step_size=step_size,
                   from_pos=from_pos, to_pos=to_pos)

    # store edge table
    #g.e.to_pickle('tmp/correlations/{}.pickle'.format(str(i).zfill(3)))
    g.e.to_pickle('{}/{}.pickle'.format(TMPpath, str(i).zfill(3)))

# computation
if __name__ == '__main__':
    os.makedirs(TMPpath, exist_ok=True)
    indices = np.arange(0, n_processes - 1)
    p = mp.Pool(n_processes)
    for _ in p.imap_unordered(create_ei, indices):
        pass

# store correlation values
files = os.listdir(TMPpath)
files.sort()
fileHDF='e'+current_time.strftime('%Y-%m-%d_%H-%M-%S')+'.h5'
store = pd.HDFStore(fileHDF, mode='w')
for f in files:
    et = pd.read_pickle('{}/{}'.format(TMPpath, f))
    store.append('e', et, format='t', data_columns=True, index=False)
store.close()

e = pd.read_hdf(fileHDF)
print(e)

#shutil.rmtree(TMPpath)
