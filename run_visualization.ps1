param(
    [ValidateSet("full-stack", "literal")]
    [string]$Mode = "full-stack",
    [int]$Samples = 320,
    [int]$Steps = 64,
    [int]$SpacetimePoints = 24,
    [string]$Output = "",
    [switch]$Fast,
    [switch]$NoShow
)

$ErrorActionPreference = 'Stop'
$projectDir = $PSScriptRoot
$python = Join-Path $projectDir '.venv-slang\Scripts\python.exe'

& $python -m pip install --disable-pip-version-check --quiet `
    -r (Join-Path $projectDir 'requirements-slang.txt')
if ($LASTEXITCODE -ne 0) {
    throw "Visualization environment setup failed with exit code $LASTEXITCODE."
}

if ($Mode -eq "literal") {
    & $python (Join-Path $projectDir 'migrate_full_slang.py') `
        --model (Join-Path $projectDir 'hybrid_model.json') `
        --output (Join-Path $projectDir 'carter_tesseract_full.slang')
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    & $python (Join-Path $projectDir 'migrate_full_slang.py') `
        --model (Join-Path $projectDir 'hybrid_model_conditioned.json') `
        --output (Join-Path $projectDir 'carter_tesseract_conditioned.slang')
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

$arguments = @(
    (Join-Path $projectDir 'visualize_tesseract.py'),
    '--mode', $Mode,
    '--samples', $Samples,
    '--steps', $Steps,
    '--spacetime-points', $SpacetimePoints
)
if ($Output) {
    $arguments += @('--output', $Output)
}
if ($Fast) {
    $arguments += '--skip-live-spacetime'
}
if ($NoShow) {
    $arguments += '--no-show'
}

& $python @arguments
exit $LASTEXITCODE
