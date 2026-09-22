
# One-shot setup for this machine: creates a venv, installs dependencies,
# and installs llama-cpp-python with the best backend available (GPU if a
# usable prebuilt wheel exists for this OS, CPU otherwise). See
# scripts/install_llama_cpp.py for the GPU/CPU decision logic.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

function Step($msg) { Write-Output "`n=== $msg ===" }

Step "Locating Python 3.12"
$pythonCmd = $null
foreach ($candidate in @("py -3.12", "python")) {
    $parts = $candidate.Split(" ")
    $exe = $parts[0]
    if (Get-Command $exe -ErrorAction SilentlyContinue) {
        $verOutput = & $exe $parts[1..($parts.Length-1)] --version 2>&1
        if ($verOutput -match "3\.1[0-9]") {
            $pythonCmd = $candidate
            break
        }
    }
}
if (-not $pythonCmd) {
    Write-Error "Python 3.10+ not found. Install it from https://www.python.org/downloads/ (check 'Add Python to PATH') and re-run."
    exit 1
}
Write-Output "Using: $pythonCmd"

Step "Creating virtual environment (venv/)"
if (-not (Test-Path "$root\venv\Scripts\python.exe")) {
    $parts = $pythonCmd.Split(" ")
    & $parts[0] $parts[1..($parts.Length-1)] -m venv venv
} else {
    Write-Output "venv already exists, skipping"
}
$venvPython = "$root\venv\Scripts\python.exe"

Step "Upgrading pip"
& $venvPython -m pip install --upgrade pip

Step "Installing argostranslate (--no-deps, avoids a sentencepiece version conflict)"
& $venvPython -m pip install --no-deps argostranslate==1.9.1

Step "Installing base requirements"
& $venvPython -m pip install -r requirements.txt

Step "Installing IndicTransToolkit (needed for IndicTrans2 to actually translate, not just load)"
& $venvPython -m pip install poetry-core
& $venvPython -m pip install "git+https://github.com/VarunGumma/indic_nlp_library"
& $venvPython -m pip install --no-build-isolation --no-deps "git+https://github.com/VarunGumma/IndicTransToolkit@0c607654e8"
if ($LASTEXITCODE -ne 0) {
    Write-Warning "IndicTransToolkit install failed (needs git on PATH) - IndicTrans2 will load but return an error on translate. Argos and GGUF will still work."
}

Step "Installing llama-cpp-python (GPU if available, CPU fallback)"
& $venvPython scripts\install_llama_cpp.py
if ($LASTEXITCODE -ne 0) {
    Write-Warning "llama-cpp-python install failed - the GGUF (IndicTrans3) backend will not be available. Argos Translate will still work."
}

Step "Done"
Write-Output "Run the app with:"
Write-Output "  venv\Scripts\python.exe P2E.py"
Write-Output "Then open http://localhost:5000"
