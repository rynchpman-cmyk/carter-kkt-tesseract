param(
    [string]$Resolutions = "16,32,64,128",
    [double]$FinalTime = 0.02,
    [double]$Cfl = 0.12,
    [int]$CorridorSamples = 129,
    [string]$Artifact = "theory33_frozen_variants.json",
    [string]$Output = "tesseract_neural_phase_riemann_results.json"
)

$ErrorActionPreference = "Stop"
python .\run_tesseract_neural_phase_riemann.py `
    --resolutions $Resolutions `
    --final-time $FinalTime `
    --cfl $Cfl `
    --corridor-samples $CorridorSamples `
    --artifact $Artifact `
    --output $Output
