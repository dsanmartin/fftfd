#!/bin/bash
# Runs the 8-mesh version with MPI, one process per mesh.
cd "$(dirname "$0")"
source /Applications/FDS/FDS6/bin/FDS6VARS.sh
export OMP_NUM_THREADS=1
mpirun -n 8 fds flame_periodic_mpi.fds
