param(
    [int]$Points = 16,
    [int]$Steps = 4,
    [double]$Dt = 1e-5,
    [string]$Checkpoint = ""
)

$arguments = @(
    "run_tesseract_nr.py",
    "--points", $Points,
    "--steps", $Steps,
    "--dt", $Dt
)
if ($Checkpoint) {
    $arguments += @("--checkpoint", $Checkpoint)
}
python @arguments
exit $LASTEXITCODE
