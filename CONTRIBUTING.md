# Contributing

Contributions are welcome, especially when they improve the mathematical
auditability, Vulkan implementation, or stability analysis.

## Development principles

1. **Keep hard decisions explicit.** Do not silently smooth parity, classifier,
   branch, or KKT active-set decisions in the production path.
2. **Differentiate the selected system.** Candidate ranking remains hard;
   continuous gradients belong to the winning KKT residual.
3. **Preserve Carter structure.** Learned constitutive behavior should descend
   from one scalar master function unless an experiment is explicitly labeled
   as an ablation.
4. **Validate derivatives externally.** Changes to a backward path need a
   centered finite-difference or equivalent numerical audit.
5. **Report margins.** A valid certificate near a switching surface is not the
   same as a robust certificate.
6. **Separate diagnostics from claims.** Continuation can demonstrate
   mechanical unrollability; only literal scale-1 runs support intrinsic
   conditioning claims.

## Setup

Reference/training dependencies:

```powershell
python -m pip install -r requirements-training.txt
```

Shader dependencies:

```powershell
python -m venv .venv-slang
.\.venv-slang\Scripts\python.exe -m pip install -r requirements-slang.txt
```

## Before submitting a change

Run the CPU-safe repository checks:

```powershell
python -m unittest discover -s tests -v
python -m py_compile *.py
```

Regenerate both Slang modules:

```powershell
.\.venv-slang\Scripts\python.exe .\migrate_full_slang.py `
    --model .\hybrid_model.json `
    --output .\carter_tesseract_full.slang

.\.venv-slang\Scripts\python.exe .\migrate_full_slang.py `
    --model .\hybrid_model_conditioned.json `
    --output .\carter_tesseract_conditioned.slang
```

For physical or differentiation changes, also run:

```powershell
python .\test_hybrid.py
.\run_full_slang.ps1
.\run_conditioned_literal.ps1 -FiniteDifferenceWeights 4
```

## Pull requests

Describe:

- the mathematical or implementation change;
- which hard signatures can change;
- validation commands and measured errors;
- effect on KKT/classifier/parity margins;
- effect on literal and continued horizons;
- whether generated Slang files and model artifacts changed.

Do not commit virtual environments, downloaded compilers, SPIR-V binaries, or
binary PyTorch checkpoints.
