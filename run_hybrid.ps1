param(
    [int] $Epochs = 30,
    [int] $Steps = 3,
    [double] $TargetZ = 0.35,
    [switch] $LearnTarget,
    [switch] $NoBoundaryCertificates
)

$ErrorActionPreference = 'Stop'
$projectDir = $PSScriptRoot

python -c "import torch; print('PyTorch', torch.__version__)"
if ($LASTEXITCODE -ne 0) {
    throw 'PyTorch is required. Install it with: python -m pip install torch'
}

$arguments = @(
    (Join-Path $projectDir 'train_hybrid.py'),
    '--epochs', $Epochs,
    '--steps', $Steps,
    '--target-z', $TargetZ
)
if ($LearnTarget) {
    $arguments += '--learn-target'
}
if ($NoBoundaryCertificates) {
    $arguments += '--no-boundary-certificates'
}

python @arguments
exit $LASTEXITCODE
