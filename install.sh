#!/usr/bin/env bash
# One-shot setup for this machine: creates a venv, installs dependencies,
# and installs llama-cpp-python with the best backend available (GPU if a
# usable prebuilt wheel exists for this OS, CPU otherwise). See
# scripts/install_llama_cpp.py for the GPU/CPU decision logic.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$root"

step() { echo; echo "=== $1 ==="; }

step "Locating Python 3.10+"
PYTHON_BIN=""
for candidate in python3.12 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        ver=$("$candidate" --version 2>&1)
        if [[ "$ver" =~ 3\.1[0-9] ]]; then
            PYTHON_BIN="$candidate"
            break
        fi
    fi
done
if [ -z "$PYTHON_BIN" ]; then
    echo "Python 3.10+ not found. Install it and re-run." >&2
    exit 1
fi
echo "Using: $PYTHON_BIN"

step "Creating virtual environment (venv/)"
if [ ! -f "$root/venv/bin/python" ]; then
    "$PYTHON_BIN" -m venv venv
else
    echo "venv already exists, skipping"
fi
VENV_PYTHON="$root/venv/bin/python"

step "Upgrading pip"
"$VENV_PYTHON" -m pip install --upgrade pip

step "Installing argostranslate (--no-deps, avoids a sentencepiece version conflict)"
"$VENV_PYTHON" -m pip install --no-deps argostranslate==1.9.1

step "Installing base requirements"
"$VENV_PYTHON" -m pip install -r requirements.txt

step "Installing IndicTransToolkit (needed for IndicTrans2 to actually translate, not just load)"
if ! "$VENV_PYTHON" -m pip install poetry-core \
    && "$VENV_PYTHON" -m pip install "git+https://github.com/VarunGumma/indic_nlp_library" \
    && "$VENV_PYTHON" -m pip install --no-build-isolation --no-deps "git+https://github.com/VarunGumma/IndicTransToolkit@0c607654e8"; then
    echo "WARNING: IndicTransToolkit install failed (needs git on PATH) - IndicTrans2 will load but return an error on translate. Argos and GGUF will still work." >&2
fi

step "Installing llama-cpp-python (GPU if available, CPU fallback)"
if ! "$VENV_PYTHON" scripts/install_llama_cpp.py; then
    echo "WARNING: llama-cpp-python install failed — the GGUF (IndicTrans3) backend won't be available. Argos Translate will still work." >&2
fi

step "Done"
echo "Run the app with:"
echo "  venv/bin/python P2E.py"
echo "Then open http://localhost:5000"
