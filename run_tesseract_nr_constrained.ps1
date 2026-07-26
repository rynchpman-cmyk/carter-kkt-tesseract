param(
    [int[]]$Resolutions = @(8, 16, 32),
    [double]$FinalTime = 1e-6,
    [int]$PicardIterations = 3,
    [double]$EllipticTolerance = 2e-8,
    [string]$Output = ""
)

$arguments = @(
    "run_tesseract_nr_constrained.py",
    "--resolutions"
) + $Resolutions + @(
    "--final-time", $FinalTime,
    "--picard-iterations", $PicardIterations,
    "--elliptic-tolerance", $EllipticTolerance
)
if ($Output) {
    $arguments += @("--output", $Output)
}
python @arguments
exit $LASTEXITCODE
