param(
    [ValidateSet("de", "en", "de-en")]
    [string]$Profile = "de-en",
    [switch]$SkipInstall,
    [switch]$RunMock,
    [switch]$RunLive
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python Launcher is required. Install Python 3.11 x64 from python.org and enable the launcher."
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    py -3.11 -m venv .venv
}

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

& $Python -m pip install --upgrade pip
if (-not $SkipInstall) {
    & $Python -m pip install -r requirements.txt
}

& $Python run.py --check --profile $Profile
& $Python -m ruff check .
& $Python -m pytest -q

if ($RunMock) {
    & $Python run.py --mock --profile $Profile
}

if ($RunLive) {
    & $Python run.py --live --profile $Profile
}
