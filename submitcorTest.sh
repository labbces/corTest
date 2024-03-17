#!/bin/bash

#$ -q all.q
#$ -cwd
#$ -pe smp 30

module load miniconda3
conda activate corTest

/usr/bin/time -v python3 computeCorrCoefParallel.py --max_cores 30 --step_size 100 --n_rows 520887 --n_cols 30 correlated_matrix_1000patterns_600000maxrows_30cols.txt
