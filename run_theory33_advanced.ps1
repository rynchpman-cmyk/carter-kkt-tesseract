param(
    [switch]$Quick,
    [string]$Output = 'theory33_advanced_results.json',
    [string]$Artifact = 'theory33_frozen_variants.json'
)

$ErrorActionPreference = 'Stop'
$arguments = @(
    (Join-Path $PSScriptRoot 'run_theory33_advanced.py'),
    '--output',
    (Join-Path $PSScriptRoot $Output),
    '--artifact',
    (Join-Path $PSScriptRoot $Artifact),
    '--data-directory',
    (Join-Path $PSScriptRoot 'data')
)
if ($Quick) {
    $arguments += '--quick'
}

python @arguments
exit $LASTEXITCODE
