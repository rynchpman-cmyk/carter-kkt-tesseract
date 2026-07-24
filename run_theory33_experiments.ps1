param(
    [switch]$Quick,
    [string]$Output = ''
)

$ErrorActionPreference = 'Stop'
$arguments = @(
    (Join-Path $PSScriptRoot 'run_theory33_experiments.py')
)
if ($Quick) {
    $arguments += '--quick'
}
if ($Output) {
    $arguments += @('--output', $Output)
}

python @arguments
exit $LASTEXITCODE
