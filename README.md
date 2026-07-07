# Offline Hindi Translator

A fully offline Flask web app for translating between Hindi and English using two engines: **Argos Translate** (fast, CPU-friendly) and **IndicTrans2** (high accuracy, requires GPU or patience on CPU).

## Project structure

```
translator/
├── P2E.py                  # Flask app
├── requirements.txt        # Python dependencies
├── models/                 # Bundled Argos Translate language packs
│   ├── en_hi/              # English → Hindi
│   └── hi_en/              # Hindi → English
├── templates/
│   └── index.html          # Web UI
└── venv/                   # Virtual environment
```

## System requirements

### Minimum (Argos Translate only)

| Component | Requirement |
|-----------|-------------|
| OS | Windows 10 / 11, Linux, macOS |
| Python | 3.10 or higher |
| RAM | 2 GB |
| Disk | 500 MB (models bundled in `models/`) |
| CPU | Any modern dual-core (2015 or newer) |
| Internet | Not required — fully offline |
| Browser | Chrome, Firefox, or Edge (any modern version) |

### Recommended (IndicTrans2 enabled)

| Component | Requirement |
|-----------|-------------|
| OS | Windows 10 / 11, Linux, macOS |
| Python | 3.10 or higher |
| RAM | 8 GB minimum · 16 GB recommended |
| Disk | 5 GB free (4 GB for IndicTrans2 model downloads + 500 MB for Argos) |
| CPU | Modern quad-core for CPU-only inference |
| GPU | NVIDIA GPU with 6 GB+ VRAM (CUDA) — strongly recommended for IndicTrans2 |
| Internet | Required once to download IndicTrans2 models from HuggingFace (~2 GB per direction) |
| Browser | Chrome, Firefox, or Edge (any modern version) |

> **Note:** Without a GPU, IndicTrans2 runs on CPU and can take **1–5 minutes per paragraph**. Argos Translate is unaffected and always runs fast on CPU.

## Requirements

- Python 3.10+
- The `venv/` virtual environment (dependencies listed in `requirements.txt`)
- For IndicTrans2: `transformers`, `torch`, `sentencepiece`, `IndicTransToolkit`, and a HuggingFace account

## Setup

```bash
# Create and activate a Python 3.12 virtual environment
py -3.12 -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# Install base dependencies
pip install -r requirements.txt

# Optional: enable IndicTrans2
pip install transformers sentencepiece torch
pip install IndicTransToolkit
```

> Note: this project currently works best with Python 3.12. The older Argos Translate dependency chain is not compatible with Python 3.13+ on Windows because SentencePiece ships binary wheels that expect 3.12.

## Running

```bash
python P2E.py
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

On first run the app loads the Argos models from `models/`. IndicTrans2 models (~1 GB each) are downloaded from HuggingFace on demand — you will be prompted for a token.

## Translation engines

| Engine | Speed | Quality | Requires |
|--------|-------|---------|----------|
| Argos Hindi | Fast | Good | Nothing extra |
| IndicTrans2 | Slow | Best | `transformers`, `torch`, HF token |

## IndicTrans2 — model details

### What it is

IndicTrans2 is an open-source neural machine translation model developed by **AI4Bharat** at IIT Madras. It is specifically built for the 22 scheduled Indian languages and is the highest-quality publicly available model for Hindi ↔ English translation.

This app uses the 1-billion parameter variants hosted on HuggingFace:

| Direction | Model ID |
|-----------|----------|
| English → Hindi | `ai4bharat/indictrans2-en-indic-1B` |
| Hindi → English | `ai4bharat/indictrans2-indic-en-1B` |

### Translation accuracy

IndicTrans2 is evaluated on the **FLORES-200** benchmark, the standard benchmark for low-resource and Indian language translation. Results for Hindi ↔ English:

| Direction | chrF++ score | BLEU score |
|-----------|-------------|------------|
| English → Hindi | ~56–58 | ~40–43 |
| Hindi → English | ~60–63 | ~47–51 |

These scores represent **state-of-the-art** quality among open-source models for Hindi. In practical terms:

- Sentence structure, grammar, and vocabulary are reliably accurate for formal and semi-formal text.
- Technical documents, business correspondence, and plain prose translate well.
- Colloquial slang, rare idioms, and highly domain-specific jargon may be imprecise.
- It significantly outperforms Argos Translate on longer sentences and complex grammar.

The model was peer-reviewed and published: *IndicTrans2: Towards High-Quality and Accessible Machine Translation Models for all 22 Scheduled Indian Languages* (AI4Bharat, 2023).

### Safety

| Property | Detail |
|----------|--------|
| **Task scope** | Pure translation only — it rephrases text, never generates new content or opinions |
| **No data collection** | Once downloaded, the model runs entirely on your machine with no internet connection required |
| **No content filtering** | The model translates whatever text is given; it does not refuse or modify content |
| **Open weights** | Model weights are publicly released under the **CC BY 4.0** license, fully auditable |
| **No telemetry** | AI4Bharat does not receive usage data when the model is run locally |
| **HuggingFace download** | Models are downloaded once (~1 GB each) from `huggingface.co` and cached locally; no further outbound calls are made |

Because it is a translation model and not a generative chat model, there is no risk of hallucinated facts, prompt injection, or unexpected content generation — it can only rephrase what you give it.

## Supported input formats (Document mode)

- **TXT** — plain text files
- **PDF** — text-based; enable OCR for scanned PDFs (requires `pymupdf` + `pytesseract` + Tesseract binary)
- **DOCX** — Word documents, preserving formatting

## OCR setup (optional)

```bash
pip install pymupdf pytesseract pillow
```

Download and install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) for Windows.
