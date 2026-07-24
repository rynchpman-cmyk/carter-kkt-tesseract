param(
    [string]$Horizons = '1,4,16,64,128',
    [int]$FiniteDifferenceWeights = 4
)

$ErrorActionPreference = 'Stop'

& (Join-Path $PSScriptRoot 'run_progressive_full.ps1') `
    -Horizons $Horizons `
    -FiniteDifferenceWeights $FiniteDifferenceWeights `
    -WeightEpsilon 0.0003 `
    -Target 0.1019493853295916 `
    -IntegrationScale 1.0 `
    -MaxLocalGain 0 `
    -Model 'hybrid_model_conditioned.json' `
    -Module 'carter_tesseract_conditioned.slang'

exit $LASTEXITCODE
