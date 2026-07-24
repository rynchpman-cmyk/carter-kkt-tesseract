param(
    [int]$Steps = 128,
    [int]$CharacteristicStride = 1,
    [int]$BasinPoints = 1025,
    [int]$BasinCharacteristicPoints = 33,
    [int]$GlobalExponentPoints = 151
)

$ErrorActionPreference = 'Stop'

python (Join-Path $PSScriptRoot 'run_theory33_hybrid.py') `
    --steps $Steps `
    --characteristic-stride $CharacteristicStride `
    --basin-points $BasinPoints `
    --basin-characteristic-points $BasinCharacteristicPoints `
    --global-exponent-points $GlobalExponentPoints

exit $LASTEXITCODE
