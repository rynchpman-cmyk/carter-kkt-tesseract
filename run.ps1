param(
    [string[]] $Values,
    [string] $Model,
    [string] $JsonOutput,
    [int] $Steps = 1
)

$ErrorActionPreference = 'Stop'
$projectDir = $PSScriptRoot
$venvPython = Join-Path $projectDir '.venv\Scripts\python.exe'
$toolDir = Join-Path $projectDir '.tools\glslang'
$validator = Join-Path $toolDir 'bin\glslangValidator.exe'
$shader = Join-Path $projectDir 'carter_tesseract_kkt.comp'
$spirv = Join-Path $projectDir 'carter_tesseract_kkt.spv'

if (-not (Test-Path -LiteralPath $venvPython)) {
    python -m venv (Join-Path $projectDir '.venv')
}

& $venvPython -m pip install --disable-pip-version-check --quiet `
    -r (Join-Path $projectDir 'requirements.txt')

if (-not (Test-Path -LiteralPath $validator)) {
    $archive = Join-Path ([System.IO.Path]::GetTempPath()) 'tesseract-glslang.zip'
    New-Item -ItemType Directory -Force -Path $toolDir | Out-Null
    Invoke-WebRequest `
        -Uri 'https://github.com/KhronosGroup/glslang/releases/download/main-tot/glslang-master-windows-Release.zip' `
        -OutFile $archive
    Expand-Archive -LiteralPath $archive -DestinationPath $toolDir -Force
}

& $validator -V --target-env vulkan1.2 -o $spirv $shader
if ($LASTEXITCODE -ne 0) {
    throw "Shader compilation failed with exit code $LASTEXITCODE."
}

$runnerArguments = @((Join-Path $projectDir 'run_tesseract.py'))
$runnerArguments += @('--steps', $Steps)
if ($Model) {
    $runnerArguments += @('--model', $Model)
}
if ($JsonOutput) {
    $runnerArguments += @('--json-output', $JsonOutput)
}
if ($Values) {
    $runnerArguments += $Values
}

& $venvPython @runnerArguments
exit $LASTEXITCODE
