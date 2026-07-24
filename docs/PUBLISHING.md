# GitHub publishing

## Recommended repository identity

- **Name:** `carter-kkt-tesseract`
- **Description:** `A hard-constrained differentiable Carter/KKT Vulkan simulator evolved from a parity-gated scalar recurrence.`
- **Topics:**
  - `automatic-differentiation`
  - `differentiable-physics`
  - `dynamical-systems`
  - `kkt`
  - `lyapunov`
  - `shader`
  - `slang`
  - `vulkan`

## First publication

After installing and authenticating the GitHub CLI:

```powershell
gh auth login
gh auth status
```

Create a private repository:

```powershell
gh repo create carter-kkt-tesseract `
    --private `
    --source . `
    --remote origin `
    --push
```

Or replace `--private` with `--public` after reviewing the repository contents.
This project is distributed under the Apache License 2.0.

Set topics:

```powershell
gh repo edit --add-topic automatic-differentiation `
    --add-topic differentiable-physics `
    --add-topic dynamical-systems `
    --add-topic kkt `
    --add-topic lyapunov `
    --add-topic shader `
    --add-topic slang `
    --add-topic vulkan
```

## Before making the repository public

1. Confirm that `LICENSE`, `NOTICE`, and `CITATION.cff` are current.
2. Confirm whether portable trained weights should be publicly redistributed.
3. Confirm the desired author identity for commits and citation metadata.
4. Re-run `python -m unittest discover -s tests -v`.
5. Re-run `.\run_conditioned_literal.ps1`.
6. Confirm the dashboard image contains no unwanted local information.

Binary checkpoints, local virtual environments, downloaded tools, and SPIR-V
binaries are already excluded by `.gitignore`.
