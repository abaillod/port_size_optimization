#!/bin/sh

#SBATCH --account=apam
#SBATCH -N 1                 # Recommended by Grace
#SBATCH --ntasks-per-node=12 # Maximum
#SBATCH --job-name=stage_II_port_size
#SBATCH --time=0-23:59       # Might be too small. Can restart from an iteration if necessary
#SBATCH --mem-per-cpu=10G     # Maximum - Ginsburg has 187G available per node

source ~/.bashrc
conda activate simsopt
module load gcc/10.2.0
module load openmpi/gcc/64/4.1.5a1
module load netcdf/gcc/64/gcc/64/4.7.4
module load lapack/gcc/64/3.9.0
module load hdf5p/1.10.7
module load netcdf-fortran/4.5.3 
module load intel-parallel-studio/2020

export OMP_NUM_THREADS=1
srun --mpi=pmix_v3 python stage_II_with_WPs.py
