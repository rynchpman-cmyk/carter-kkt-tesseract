param(
    [string]$Resolutions = "16,32,64,128",
    [double]$FinalTime = 0.02,
    [double]$Cfl = 0.15,
    [string]$Output = "tesseract_carter_riemann_results.json"
)

$ErrorActionPreference = "Stop"
python .\run_tesseract_carter_riemann.py `
    --resolutions $Resolutions `
    --final-time $FinalTime `
    --cfl $Cfl `
    --output $Output
