# NASA C-MAPSS turbofan engine degradation data

These are the original, unmodified files of the C-MAPSS dataset, included so the
project runs straight after cloning. `NASA_readme.txt` is NASA's own description
of the files and column layout.

| Subset | Train engines | Test engines | Operating conditions | Fault modes |
|---|---|---|---|---|
| FD001 | 100 | 100 | 1 (sea level) | HPC degradation |
| FD002 | 260 | 259 | 6 | HPC degradation |
| FD003 | 100 | 100 | 1 (sea level) | HPC and fan degradation |
| FD004 | 248 | 249 | 6 | HPC and fan degradation |

- `train_FD00x.txt`: run-to-failure trajectories
- `test_FD00x.txt`: trajectories that stop some time before failure
- `RUL_FD00x.txt`: true remaining useful life at the end of each test trajectory

**Source:** NASA Prognostics Center of Excellence (PCoE) Data Set Repository, "Turbofan Engine
Degradation Simulation Data Set".

**Reference:** A. Saxena, K. Goebel, D. Simon and N. Eklund, "Damage Propagation Modeling for
Aircraft Engine Run-to-Failure Simulation", Proceedings of the 1st International Conference on
Prognostics and Health Management (PHM08), Denver, CO, 2008.
