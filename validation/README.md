# Validation: pycora vs CORA MATLAB

Compares pycora's reach output against the original CORA MATLAB on the same
KS bicycle problem (curve, dt=0.1, 10 steps, taylor=6, zonotope_order=50).

## Setup

- Windows MATLAB with CORA installed via Add-On Explorer:
  Home tab → Add-Ons → search "CORA" → Install.
  (Identical to the GitHub repo, just auto-managed.)
- WSL with this project at the standard project path.

## Run on Windows MATLAB

Open MATLAB on Windows. From the MATLAB command window:

```matlab
cd('\\wsl.localhost\Ubuntu\home\larisa\Projects\commonroad-frenetix-project\validation')
run('cora_ks_reach.m')
```

This writes `cora_outputs/ks_curve_reach_bounds.csv`.

If MATLAB can't write to the WSL path directly, change `output_dir` in the
script to `'C:\temp\pycora_validation'` and copy the CSV to
`validation/cora_outputs/` afterwards.

## Run pycora (in WSL)

```bash
PYTHONPATH=src python validation/pycora_ks_reach.py
```

This writes `pycora_outputs/ks_curve_reach_bounds.csv`.

## Compare

```bash
PYTHONPATH=src python validation/compare_cora_pycora.py
```

Prints per-step absolute differences in lower/upper bounds for each state
dimension, and a side-by-side final-step comparison.

## Calling MATLAB from WSL (alternative)

If you want WSL to invoke Windows MATLAB directly:

```bash
"/mnt/c/Program Files/MATLAB/R2024a/bin/matlab.exe" -batch \
  "cd('\\\\wsl.localhost\\Ubuntu\\home\\larisa\\Projects\\commonroad-frenetix-project\\validation'); \
   run('cora_ks_reach.m')"
```

(Adjust the MATLAB version. CORA is on the path automatically if installed via Add-Ons.)

## What "matches" means

Because pycora's port simplifies several things (no matrix zonotopes, no
adaptive step size, no set splitting), some numerical drift is expected.
Targets:
- **Centers** should match within ~1e-6 (deterministic propagation).
- **Bounds (widths)** may differ by a few percent due to different
  reduction strategies and the order in which sets are enclosed/merged.

If centers drift > 1e-3 or widths differ by > 50%, that's a bug, not just
implementation drift.
