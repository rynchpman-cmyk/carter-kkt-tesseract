param(
    [int[]]$Resolutions = @(8, 16, 32),
    [double]$FinalTime = 1e-5,
    [string]$Output = ""
)

$arguments = @(
    "run_tesseract_nr_convergence.py",
    "--resolutions"
) + $Resolutions + @(
    "--final-time", $FinalTime
)
if ($Output) {
    $arguments += @("--output", $Output)
}
python @arguments
exit $LASTEXITCODE
