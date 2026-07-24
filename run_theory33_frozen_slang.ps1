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

python (Join-Path $projectDir 'export_theory33_frozen_slang.py') `
    --artifact (Join-Path $projectDir 'theory33_frozen_variants.json') `
    --base (Join-Path $projectDir 'theory33_hybrid.slang') `
    --output (Join-Path $projectDir 'theory33_qualified_frozen.slang')
if ($LASTEXITCODE -ne 0) {
    throw "Frozen Theory 3.3 export failed with exit code $LASTEXITCODE."
}

& $python (Join-Path $projectDir 'test_theory33_frozen_slang.py')
exit $LASTEXITCODE
