# FDS case for the fire replay

Input for the Fire Dynamics Simulator (FDS, tested with version 6.11.0) that produces the density and
pressure fields used in the section "Replay of a fire simulation" of the paper.

**Case:** a 60 kW propane burner (0.30 m x 0.30 m) on the floor of a 1.92 m cube, 64 x 64 x 64 cells
of 3 cm, periodic lateral boundaries, open top, solid floor, fixed time step of 2 ms, 6.4 s of
simulated time.

| File | Meshes | Run with | Notes |
|---|---|---|---|
| `flame_periodic_mpi.fds` | 8 meshes of 32^3 | `./run_mpi.sh` (8 MPI processes) | Used for the results in the paper. |
| `omp/flame_periodic.fds` | 1 mesh | `./run.sh` (4 OpenMP threads) | Same case on one mesh. |

The run scripts assume FDS is installed in `/Applications/FDS/FDS6` (macOS bundle); edit the paths
for other systems. On an Apple M4 the MPI version takes about 10 minutes.

**Output.** Cell-centred 3D fields of density, pressure and temperature at t = 2 s and 4 s and at
every time step from 6.0 s to 6.4 s (about 0.7 GB), plus 2D slices every 0.02 s for Smokeview
(planes x = 0.96 m, y = 0.96 m and z = 0.30 m). The output is not tracked by git.

**Replay.** From the repository root:

```bash
python fds_replay.py fds        # writes paper/data/fds_replay.json
```

Two details of the input that are easy to get wrong:

- 3D slices are scheduled with `RAMP_SL3D`; `DT_SLCF` and `RAMP_SLCF` only affect 2D slices.
- With more than one mesh, periodic boundaries must be declared by plane (`PBX`, `PBY`), not by mesh face (`MB`).

In a multi-mesh run FDS matches the pressure across mesh interfaces only to a tolerance. The
single-mesh case avoids this.
