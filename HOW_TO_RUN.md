# How to Run — Setup Guide (16 GB RAM Machine)

## 1. Install Python

Download and install **Python 3.10 or higher** from https://www.python.org/downloads/

During installation, check **"Add Python to PATH"**.

Verify after install:
```
python --version
```

---

## 2. Copy the project folder

Copy the entire `translator/` folder to the new machine. It already contains:
- `models/` — Argos Hindi language packs (no download needed)
- `P2E.py` — the app
- `requirements.txt` — dependencies list

---

## 3. Create a virtual environment

Open a terminal inside the `translator/` folder and run:

```bash
py -3.12 -m venv venv
```

Activate it:

```bash
# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

You should see `(venv)` appear at the start of your terminal prompt.

---

## 4. Install base dependencies

```bash
pip install -r requirements.txt
```

This installs Flask and Argos Translate. Argos Hindi works immediately after this — no internet needed.

---

## 5. Install IndicTrans2 dependencies (recommended for 16 GB RAM)

With 16 GB RAM, IndicTrans2 runs well. The app **automatically uses the GPU if one is available** — no extra configuration needed. On CPU it still works but is slower.

> **Important version note:** IndicTrans2 requires `transformers 4.x`. The newer `transformers 5.x` removed a module the model depends on (`transformers.onnx`) and will cause a startup error. Always install `transformers` with the `<5.0` constraint below.

### If the machine has an NVIDIA GPU (recommended)

> **GPU requirement:** PyTorch 2.x requires CUDA compute capability **7.5 or higher** (Turing architecture — RTX 20xx, Quadro RTX, T4, A100, etc.). Older GPUs like the Quadro M5000, GTX 10xx, Quadro P series (CC 5.x / 6.x) are **not supported** — the app will automatically fall back to CPU for those.

Go to https://pytorch.org/get-started/locally/ and generate the install command for your CUDA version. Example for CUDA 12.1:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install "transformers>=4.40,<5.0" sentencepiece
pip install IndicTransToolkit
```

At startup the terminal will print:
```
Using device: cuda
```

### If the machine has no GPU (CPU only)

```bash
pip install torch
pip install "transformers>=4.40,<5.0" sentencepiece
pip install IndicTransToolkit
```

At startup the terminal will print:
```
Using device: cpu
```

> **How GPU detection works:** the app runs `torch.cuda.is_available()` at startup. If a CUDA-capable GPU is found, all IndicTrans2 inference runs on it automatically using float16 precision (faster + lower memory). No setting to change.

### Already installed transformers 5.x by mistake?

Run this to downgrade — pip will also fix tokenizers automatically:

```bash
pip install "transformers>=4.40,<5.0"
```

---

## 6. Run the app

```bash
python P2E.py
```

The terminal will show:

```
Initializing Argos Translate for Hindi...
Starting Flask server...
Open your browser and go to: http://localhost:5000
```

Open your browser and go to: **http://localhost:5000**

---

## 7. First-time IndicTrans2 model download

The first time you select the **IndicTrans2** engine and translate, the terminal will prompt:

```
--- HuggingFace Login Required for IndicTrans2 ---
Enter your HuggingFace token:
```

1. Create a free account at https://huggingface.co
2. Go to **Settings → Access Tokens → New token** (read access is enough)
3. Paste the token into the terminal

The models will download (~2 GB per direction, ~4 GB total) and are cached locally. You will not be prompted again.

---

## 8. Activate and run on future sessions

Every time you want to use the app:

```bash
# Inside the translator/ folder
venv\Scripts\activate       # Windows
python P2E.py
```

Then open **http://localhost:5000** in your browser.

---

## Expected performance on 16 GB RAM (CPU only)

| Engine | Speed |
|--------|-------|
| Argos (Hindi ↔ English) | Instant (~1–2 seconds per paragraph) |
| IndicTrans2 — CPU only | ~1–3 minutes per paragraph |
| IndicTrans2 — with NVIDIA GPU | ~3–10 seconds per paragraph |

---

## Troubleshooting

**`python` not found**
→ Python is not in PATH. Re-install and check "Add to PATH", or use `python3` instead.

**`pip install torch` fails or is very slow**
→ Use the selector at https://pytorch.org/get-started/locally/ for the right command.

**`No module named flask`**
→ The venv is not activated. Run `venv\Scripts\activate` first.

**IndicTrans2 shows "Error: IndicTrans2 not loaded"**
→ The extra packages are not installed. Run step 5 above and restart the app.

**Port 5000 already in use**
→ Another app is using port 5000. Either stop it, or change the port in `P2E.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001, use_reloader=False)
```
Then open http://localhost:5001
