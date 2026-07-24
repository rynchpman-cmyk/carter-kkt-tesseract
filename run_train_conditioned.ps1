param(
    [int]$CaptureEpochs = 400,
    [int]$CurriculumEpochs = 20,
    [int]$ValidationSteps = 128
)

$ErrorActionPreference = 'Stop'
$projectDir = $PSScriptRoot

$arguments = @(
    (Join-Path $projectDir 'train_lyapunov_conditioned.py'),
    '--capture-epochs', $CaptureEpochs,
    '--curriculum-epochs', $CurriculumEpochs,
    '--validation-steps', $ValidationSteps
)

python @arguments
exit $LASTEXITCODE
