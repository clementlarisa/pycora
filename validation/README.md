# Validation: pycora vs CORA MATLAB

We can compare pycora's reach output against the original CORA MATLAB on the same
KS bicycle problem (curve, dt=0.1, 10 steps, taylor=6, zonotope_order=50).

## Setup

- Windows MATLAB with CORA installed via Add-On Explorer:
  Home tab > Add-Ons > search "CORA" > Install.
- WSL with this project at the standard project path.

## Run on Windows MATLAB

Open MATLAB on Windows. From the MATLAB command window:

```matlab
run('cora_ks_reach.m')
```

=> `cora_outputs/ks_curve_reach_bounds.csv`.

## Run pycora

```bash
PYTHONPATH=src python validation/pycora_ks_reach.py
```
=> `pycora_outputs/ks_curve_reach_bounds.csv`.

## Compare

```bash
PYTHONPATH=src python validation/compare_cora_pycora.py
```

=> prints per-step absolute differences in lower/upper bounds for each state
dimension, and a side-by-side comparison.

## Assumptions

Because the python port simplifies several things (no matrix zonotopes, no
adaptive step size, no set splitting), some numerical drift is expected
Targets:
- **Centers** should match within ~1e-6 (deterministic propagation).
- **Bounds (widths)** may differ by a few percent due to different
  reduction strategies and the order in which sets are enclosed/merged.

A successful port should have at most centers drift <= 1e-3 or widths differ by <= 50%