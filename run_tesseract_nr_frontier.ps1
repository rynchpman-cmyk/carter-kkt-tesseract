param(
    [int]$HorizonSteps = 128,
    [string]$Artifact = "theory33_frozen_variants.json",
    [string]$Output = "tesseract_nr_frontier_results.json",
    [switch]$Skip3D
)

$arguments = @(
    "run_tesseract_nr_frontier.py",
    "--horizon-steps", $HorizonSteps,
    "--artifact", $Artifact,
    "--output", $Output
)
if ($Skip3D) {
    $arguments += "--skip-3d"
}
python @arguments
exit $LASTEXITCODE
