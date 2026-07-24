# Reproducibility

## Supported environment

The validated environment is:

- Windows 10/11 PowerShell;
- Python 3.11+;
- Vulkan-capable GPU and current driver;
- PyTorch `2.11.0+cpu` for reference training;
- SlangPy `0.43.1`;
- NumPy `2.5.1` in the Slang environment;
- Matplotlib `3.10.0`.

The PowerShell runners create or reuse local virtual environments and install
pinned shader-side dependencies.

## Environment setup

Training/reference environment:

```powershell
python -m pip install -r requirements-training.txt
```

Slang/Vulkan environment:

```powershell
python -m venv .venv-slang
.\.venv-slang\Scripts\python.exe -m pip install -r requirements-slang.txt
```

The original GLSL runner creates `.venv` and downloads
`glslangValidator` into `.tools` on first use.

## Deterministic shader generation

Generate the baseline:

```powershell
.\.venv-slang\Scripts\python.exe .\migrate_full_slang.py `
    --model .\hybrid_model.json `
    --output .\carter_tesseract_full.slang
```

Generate the conditioned module:

```powershell
.\.venv-slang\Scripts\python.exe .\migrate_full_slang.py `
    --model .\hybrid_model_conditioned.json `
    --output .\carter_tesseract_conditioned.slang
```

The generated files are tracked. CI regenerates both and fails if the source
tree changes.

## Validation tiers

### 1. Artifact/source checks

```powershell
python -m unittest discover -s tests -v
python -m py_compile *.py
```

These checks do not require a GPU.

### 2. PyTorch differentiability

```powershell
python .\test_hybrid.py
```

Checks finite recurrence loss, exact KKT feasibility, and a local derivative
against centered finite differences.

### 3. Theory 3.3 physical hybrid

```powershell
.\run_theory33_hybrid.ps1
```

Runs 128 literal steps with KKT, timelike-current, convexity, causal-cone,
principal-symbol, and Lyapunov checks. It additionally evolves a 1,025-point
initialization grid over `[-1, 1]` and samples both characteristic branches
over the resulting enclosure. Finally, it sends signed log-spaced inputs
through magnitude `1e300` through the global absorbing gate and verifies
finite evolution, zero outer sensitivity, and physical validity. This tier is
CPU-only.

Regenerate and verify the compact controller coefficients with:

```powershell
python .\fit_theory33_basin.py
```

Run the isolated nonsmooth and neural-constitutive experiment ladder with:

```powershell
.\run_theory33_experiments.ps1 `
    -Output .\theory33_experiment_results.json
```

The full run trains all experimental modules and then applies the same
128-step basin, physical, characteristic, and global-safety gates. Use
`-Quick` for a shorter development fit with unchanged acceptance audits.
Measurements and limitations are documented in
[the experiment report](../THEORY33_EXPERIMENT_REPORT.md).

Generate the independent kinetic EOS, transport, and noisy phase tables:

```powershell
python .\generate_theory33_kinetic_data.py `
    --output-directory .\data
```

Run the full frozen-constitutive qualification:

```powershell
.\run_theory33_advanced.ps1
```

This trains from the frozen tables, uses held-out trajectories, bootstraps
phase uncertainty, compares multi-step event-gradient estimators, continues
the Carter scale against explicit margins, writes
`theory33_frozen_variants.json`, and then replays the artifact without
fitting. Use `-Quick` for development resolution.

Export only the qualified artifact and validate it on Vulkan:

```powershell
.\run_theory33_frozen_slang.ps1
```

See the
[advanced qualification report](../THEORY33_ADVANCED_REPORT.md) for hashes,
acceptance thresholds, results, and limitations.

### 4. Theory 3.3 native Slang closure

```powershell
.\run_theory33_slang.ps1
```

Checks the analytic M1 constitutive values and native reverse gradient against
an independent float32 implementation on Vulkan.

### 5. GLSL/PyTorch parity

```powershell
python .\compare_backends.py --steps 2
```

Checks recurrence values, active sets, and classifier masks.

### 6. Native Slang reverse

```powershell
.\run_full_slang.ps1
```

Checks compact/full primal equality, nonzero adjoints, all 1,287 weights, and
finite-difference agreement.

### 7. Progressive full physics

```powershell
.\run_progressive_full.ps1
```

Runs continuation horizons and whole-trajectory BPTT audits.

### 8. Conditioned literal milestone

```powershell
.\run_conditioned_literal.ps1
```

Requires:

- scale `1.0`;
- no gain control;
- valid KKT certificates;
- healthy margins;
- non-increasing Lyapunov value;
- convergence at horizons 64 and 128.

## Retraining

```powershell
.\run_train_conditioned.ps1
```

The checkpoint is written locally as `hybrid_tesseract_conditioned.pt` and is
ignored by Git. The portable `hybrid_model_conditioned.json` is tracked.

Because GPU kernels use float32 and training uses float64 PyTorch, final digits
can differ while remaining inside the reported error bounds.

## Known platform warning

SlangPy can print a D3D12 Agility SDK warning when the Python executable and
SDK live on different drives. The project explicitly creates a Vulkan device;
the warning does not affect the validated Vulkan path.
