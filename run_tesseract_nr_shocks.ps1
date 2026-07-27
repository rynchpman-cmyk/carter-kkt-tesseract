param(
    [int]$PhasePoints = 8,
    [int]$PositivityPoints = 16,
    [string]$Output = "tesseract_nr_shock_results.json"
)

$ErrorActionPreference = "Stop"
python .\run_tesseract_nr_shocks.py `
    --phase-points $PhasePoints `
    --positivity-points $PositivityPoints `
    --output $Output
