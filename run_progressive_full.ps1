param(
    [string]$Horizons = '1,2,4,8,16,32',
    [double[]]$Initial = @(-0.8, -0.2, 0.2, 0.8),
    [double]$Target = 0.2,
    [int]$FiniteDifferenceWeights = 4,
    [double]$IntegrationScale = 0.05,
    [double]$MaxLocalGain = 1.1,
    [double]$WeightEpsilon = 0.03,
    [string]$Model = 'hybrid_model.json',
    [string]$Module = 'carter_tesseract_full.slang'
)

$ErrorActionPreference = 'Stop'
$projectDir = $PSScriptRoot
$python = Join-Path $projectDir '.venv-slang\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $python)) {
    python -m venv (Join-Path $projectDir '.venv-slang')
}

& $python -m pip install --disable-pip-version-check --quiet `
    -r (Join-Path $projectDir 'requirements-slang.txt')
if ($LASTEXITCODE -ne 0) {
    throw "SlangPy environment setup failed with exit code $LASTEXITCODE."
}

$modelPath = Join-Path $projectDir $Model
$modulePath = Join-Path $projectDir $Module

& $python (Join-Path $projectDir 'migrate_full_slang.py') `
    --model $modelPath `
    --output $modulePath
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$culture = [System.Globalization.CultureInfo]::InvariantCulture
$arguments = @(
    (Join-Path $projectDir 'progressive_full_unroll.py'),
    '--horizons',
    $Horizons,
    '--target',
    $Target.ToString($culture),
    '--fd-weights',
    $FiniteDifferenceWeights.ToString($culture),
    '--weight-epsilon',
    $WeightEpsilon.ToString($culture),
    '--integration-scale',
    $IntegrationScale.ToString($culture),
    '--max-local-gain',
    $MaxLocalGain.ToString($culture),
    '--model',
    $modelPath,
    '--module',
    $modulePath,
    '--initial'
)
$arguments += $Initial | ForEach-Object { $_.ToString($culture) }

& $python @arguments
exit $LASTEXITCODE
