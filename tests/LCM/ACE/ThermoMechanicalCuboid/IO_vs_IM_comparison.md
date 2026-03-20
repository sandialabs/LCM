# I/O vs In-Memory ACE Solver Comparison

## Test: ThermoMechanicalCuboid (serial, Clang 15.0.7)
- Mesh: cuboid_denudation.g (128 hex8 elements, 225 nodes, 2 blocks)
- Thermal: Forward Euler, Neumann BC (geothermal heat flux) + time-dependent Dirichlet
- Mechanical: TrapezoidRule, J2 Erosion with Disable Erosion
- Coupling: ACE_Ice_Saturation transfer only (IM), full exodus restart (I/O)
- Time: 0 → 64800 sec, dt = 900 sec, 72 steps

## Thermal Solution Average (Response[1]) Comparison

| Step | Time [s] | IM | I/O | Rel Diff [%] |
|-----:|----------|-----------------|-----------------|------------:|
| 0 | 0 | 2.70000000e+02 | 2.70000000e+02 | 0.000000 |
| 1 | 900 | 2.70161111e+02 | 2.70161134e+02 | -0.000009 |
| 2 | 1800 | 2.70322222e+02 | 2.70325356e+02 | -0.001159 |
| 3 | 2700 | 2.70483333e+02 | 2.70492575e+02 | -0.003417 |
| 4 | 3600 | 2.70644444e+02 | 2.70662704e+02 | -0.006746 |
| 5 | 4500 | 2.70805556e+02 | 2.70835656e+02 | -0.011114 |
| 10 | 9000 | 2.71611111e+02 | 2.71732692e+02 | -0.044743 |
| 15 | 13500 | 2.72416667e+02 | 2.72679441e+02 | -0.096367 |
| 20 | 18000 | 2.73222222e+02 | 2.73650742e+02 | -0.156594 |
| 25 | 22500 | 2.74027778e+02 | 2.74592461e+02 | -0.205644 |
| 30 | 27000 | 2.74833333e+02 | 2.75506479e+02 | -0.244330 |
| 35 | 31500 | 2.75638889e+02 | 2.76397145e+02 | -0.274336 |
| 40 | 36000 | 2.76444444e+02 | 2.77263512e+02 | -0.295411 |
| 45 | 40500 | 2.76442200e+02 | 2.77308945e+02 | -0.312556 |
| 50 | 45000 | 2.76439955e+02 | 2.77343890e+02 | -0.325926 |
| 55 | 49500 | 2.76437710e+02 | 2.77365091e+02 | -0.334354 |
| 60 | 54000 | 2.76435466e+02 | 2.77380176e+02 | -0.340583 |
| 65 | 58500 | 2.76433221e+02 | 2.77392363e+02 | -0.345771 |
| 71 | 63900 | 2.76430527e+02 | 2.77404858e+02 | -0.351231 |

## Observations

- Divergence stabilizes and is bounded (~0.35%).
- IM temperature is systematically lower.
- The rate of divergence decreases with time (approaching asymptote).
- Consistent with different coupling fidelity: I/O restarts with full
  mesh state; IM transfers only ACE_Ice_Saturation.

## Wall Times (serial, Clang 15.0.7, Rigel)

- I/O solver: 18.60 sec
- IM solver:  14.81 sec
- Speedup:    1.26x (20% faster)
