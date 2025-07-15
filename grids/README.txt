VULCAN disequilibrium chemistry grids

Must copy these grids into your inputs/chemistry_grids directory

Current supported grids:
===VULCAN_Grid1.1===
=Variable parameters=
   log_kappa_IR               [-3.5, -2.5], step=0.25
   T_equ (K)                  [700, 1100], step=100
   C/O                        (0.2, 1, 1.5, 2)
   log_met                    [-1, 4], step=1
=Fixed parameters=
   log_gamma                  -1.0
   T_int (K)                  358
   log_Kzz (cm/s**2)          10.5

===VULCAN_Grid2.0===
=Variable parameters=
    log_kappa_IR            [-5.5, -1.5], step=1
    log_gamma               [-3.0, 0.0], step=1
    T_equ (K)               [650, 1050], step=100
    C/O                     (0.2, 0.5, 1)
    log_met                 [0, 2] step=1
    log_Kzz (cm/s**2)       [7.5, 11.5], step=1
=Fixed parameters=
    T_int (K)               358