param(
    [int]$Samples = 320,
    [int]$Steps = 64,
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

$arguments = @(
    (Join-Path $projectDir 'visualize_tesseract.py'),
    '--samples', $Samples,
    '--steps', $Steps
)
if ($NoShow) {
    $arguments += '--no-show'
}

& $python @arguments
exit $LASTEXITCODE
