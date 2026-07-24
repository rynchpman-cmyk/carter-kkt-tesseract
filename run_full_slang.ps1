param(
    [switch]$PhysicalOnly
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

& $python (Join-Path $projectDir 'migrate_full_slang.py')
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$testArguments = @(
    (Join-Path $projectDir 'test_full_slang_recurrence.py')
)
if ($PhysicalOnly) {
    $testArguments += '--physical-only'
}

& $python @testArguments
exit $LASTEXITCODE
