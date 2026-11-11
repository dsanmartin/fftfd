#!/bin/bash
# Runs the single-mesh case in omp/. One mesh, so the parallelism is OpenMP;
# 4 threads (the performance cores) was the fastest setting on the Apple M4.
cd "$(dirname "$0")/omp"
source /Applications/FDS/FDS6/bin/FDS6VARS.sh 2>/dev/null
export OMP_NUM_THREADS=4
/Applications/FDS/FDS6/bin/fds_openmp flame_periodic.fds
