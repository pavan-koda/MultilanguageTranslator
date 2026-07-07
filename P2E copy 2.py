from flask import Flask, render_template, request, jsonify, send_file, Response, stream_with_context
import argostranslate.package
import argostranslate.translate
import argostranslate.settings
import os
import io
import json
import base64
import re
import copy
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

# Point argostranslate at its own subfolder so other model dirs don't interfere
_argos_dir = Path(__file__).parent / "models" / "argos_packages"
_argos_dir.mkdir(parents=True, exist_ok=True)
argostranslate.settings.package_dirs = [_argos_dir]

try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    import fitz  # pymupdf
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image as PilImage
    PYTESSERACT_AVAILABLE = True
    # Auto-detect Tesseract binary on Windows
    import os as _os
    for _tp in [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(
            _os.environ.get('USERNAME', '')),
    ]:
        if _os.path.exists(_tp):
            pytesseract.pytesseract.tesseract_cmd = _tp
            break
except ImportError:
    PYTESSERACT_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from IndicTransToolkit.processor import IndicProcessor
    INDIC_TOOLKIT_AVAILABLE = True
except ImportError:
    INDIC_TOOLKIT_AVAILABLE = False

app = Flask(__name__)

indic_tokenizers = {}
indic_models = {}
indic_processor = None
_glossary = {}       # effective glossary: base + custom additions - deletions
_base_glossary = {}  # loaded from hindi data.csv, never mutated at runtime
_overrides = {'added': {}, 'deleted': []}  # persisted user edits

OVERRIDES_PATH = Path(__file__).parent / "glossary_overrides.json"

# ── IndicTrans3-beta (Gemma-3 based, document-level) ────────────────────────
_it3_model      = None   # transformers backend
_it3_tokenizer  = None
_it3_loading    = False
_it3_load_error = None

_it3_gguf_model   = None  # GGUF backend (llama-cpp-python)
_it3_gguf_loading = False
_it3_gguf_error   = None

IT3_MODEL_ID  = "ai4bharat/IndicTrans3-beta"
IT3_GGUF_REPO = "ai4bharat/IndicTrans3-beta-GGUF"   # HF repo for pre-built GGUF
IT3_GGUF_FILE = "indictrans3-beta.Q4_K_M.gguf"      # preferred quantisation
IT3_GGUF_DIR  = Path(__file__).parent / "models" / "it3_gguf"

IT3_LANG_NAMES = {
    'en': 'English',
    'hi': 'Hindi (hin_Deva)',
}


# ── Backend selection ────────────────────────────────────────────────────────

def _use_gguf_backend():
    """Use GGUF when the GPU is too old for bitsandbytes (CC < 7.5) or absent.
    GGUF via llama.cpp supports CUDA from CC 3.5+, so old GPUs like K2200 work."""
    if not TRANSFORMERS_AVAILABLE:
        return True
    try:
        import torch
        if not torch.cuda.is_available():
            return True   # no GPU at all — GGUF CPU is faster than PyTorch CPU
        cc = torch.cuda.get_device_capability(0)
        return cc < (7, 5)  # bitsandbytes needs Turing+; below that use GGUF
    except Exception:
        return True


# ── Status ───────────────────────────────────────────────────────────────────

def _it3_status():
    if _use_gguf_backend():
        return {
            'loaded':  _it3_gguf_model is not None,
            'loading': _it3_gguf_loading,
            'error':   _it3_gguf_error,
            'backend': 'gguf',
        }
    return {
        'loaded':  _it3_model is not None,
        'loading': _it3_loading,
        'error':   _it3_load_error,
        'backend': 'transformers',
    }


# ── Transformers backend ─────────────────────────────────────────────────────

def _load_it3_model():
    """Lazy-load IndicTrans3-beta via HuggingFace Transformers (CC ≥ 7.5 only)."""
    global _it3_model, _it3_tokenizer, _it3_loading, _it3_load_error
    if _it3_model is not None:
        return True
    if _it3_loading:
        return False
    _it3_loading    = True
    _it3_load_error = None
    try:
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers / torch not installed.")
        import torch
        print(f"\nLoading {IT3_MODEL_ID} — first run downloads ~8 GB …")
        _it3_tokenizer = AutoTokenizer.from_pretrained(IT3_MODEL_ID, trust_remote_code=True)
        load_kwargs = dict(trust_remote_code=True, device_map='auto')
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            try:
                from transformers import BitsAndBytesConfig
                if vram_gb < 12:
                    print(f"VRAM={vram_gb:.1f} GB — loading IT3 in 4-bit (bitsandbytes)")
                    load_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True)
                elif vram_gb < 18:
                    print(f"VRAM={vram_gb:.1f} GB — loading IT3 in 8-bit (bitsandbytes)")
                    load_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
                else:
                    load_kwargs['dtype'] = torch.float16
            except ImportError:
                print("bitsandbytes not installed — using float16 + device_map offloading")
                load_kwargs['dtype'] = torch.float16
        else:
            load_kwargs['dtype'] = torch.float32
        _it3_model = AutoModelForCausalLM.from_pretrained(IT3_MODEL_ID, **load_kwargs)
        _it3_model.eval()
        print(f"{IT3_MODEL_ID} loaded successfully.")
        return True
    except Exception as exc:
        _it3_load_error = str(exc)
        print(f"IT3 load error: {exc}")
        return False
    finally:
        _it3_loading = False


# ── GGUF backend ─────────────────────────────────────────────────────────────

def _find_local_gguf():
    """Return path to an existing local GGUF file, or None."""
    IT3_GGUF_DIR.mkdir(parents=True, exist_ok=True)
    preferred = IT3_GGUF_DIR / IT3_GGUF_FILE
    if preferred.exists():
        return preferred
    candidates = sorted(IT3_GGUF_DIR.glob("*.gguf"))
    return candidates[0] if candidates else None


def _download_gguf():
    """Try to download a GGUF file from HuggingFace. Returns path or None."""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        print(f"Looking for GGUF files in {IT3_GGUF_REPO} ...")
        try:
            files = list(list_repo_files(IT3_GGUF_REPO))
        except Exception:
            print(f"  Repo '{IT3_GGUF_REPO}' not found — place a .gguf file manually in {IT3_GGUF_DIR}")
            return None
        gguf_files = [f for f in files if f.endswith('.gguf')]
        if not gguf_files:
            print(f"  No .gguf files found in {IT3_GGUF_REPO}")
            return None
        target = next((f for f in gguf_files if 'Q4_K_M' in f), gguf_files[0])
        print(f"  Downloading {target} ...")
        path = hf_hub_download(
            repo_id=IT3_GGUF_REPO,
            filename=target,
            local_dir=str(IT3_GGUF_DIR),
        )
        return Path(path)
    except Exception as e:
        print(f"  GGUF download failed: {e}")
        return None


def _load_it3_gguf():
    """Load IT3 as a GGUF model via llama-cpp-python. Works on CC 3.5+ (K2200 included)."""
    global _it3_gguf_model, _it3_gguf_loading, _it3_gguf_error
    if _it3_gguf_model is not None:
        return True
    if _it3_gguf_loading:
        return False
    _it3_gguf_loading = True
    _it3_gguf_error   = None
    try:
        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python not installed.\n"
                "  CPU:  pip install llama-cpp-python\n"
                "  CUDA: pip install llama-cpp-python "
                "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124"
            )

        gguf_path = _find_local_gguf()
        if gguf_path is None:
            print("No local GGUF found — attempting HuggingFace download ...")
            gguf_path = _download_gguf()

        if gguf_path is None:
            raise RuntimeError(
                f"No GGUF file available.\n"
                f"Option 1 — Place any IT3 .gguf file in: {IT3_GGUF_DIR}\n"
                f"Option 2 — Convert the HF model:\n"
                f"  git clone https://github.com/ggerganov/llama.cpp\n"
                f"  python llama.cpp/convert_hf_to_gguf.py <IT3-hf-folder> "
                f"--outtype q4_k_m --outfile {IT3_GGUF_DIR / IT3_GGUF_FILE}"
            )

        # Calculate how many transformer layers fit in available VRAM
        n_gpu_layers = 0
        try:
            import torch
            if torch.cuda.is_available():
                vram_gb   = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                free_gb   = vram_gb * 0.80          # leave 20% headroom
                mb_per_layer = 55                    # ~55 MB per layer for Q4_K_M 4B model
                n_gpu_layers = min(28, int(free_gb * 1024 / mb_per_layer))
                cc = torch.cuda.get_device_capability(0)
                print(f"GGUF: GPU CC={cc[0]}.{cc[1]}, VRAM={vram_gb:.1f} GB "
                      f"— offloading {n_gpu_layers}/28 layers")
        except Exception:
            pass

        n_threads = max(1, (os.cpu_count() or 4) // 2)  # half cores, keep system usable
        print(f"Loading GGUF: {gguf_path.name}  "
              f"(gpu_layers={n_gpu_layers}, threads={n_threads})")
        _it3_gguf_model = Llama(
            model_path   = str(gguf_path),
            n_ctx        = 4096,
            n_gpu_layers = n_gpu_layers,
            n_threads    = n_threads,
            verbose      = False,
        )
        print("IT3 GGUF model loaded successfully.")
        return True
    except Exception as exc:
        _it3_gguf_error = str(exc)
        print(f"IT3 GGUF load error: {exc}")
        return False
    finally:
        _it3_gguf_loading = False


def _translate_with_it3_gguf(text, from_lang, to_lang):
    """Translate via GGUF backend — works on K2200 (CC 5.0) and any newer GPU."""
    global _it3_gguf_model
    if _it3_gguf_model is None:
        if not _load_it3_gguf():
            return f"Error: could not load IT3 GGUF — {_it3_gguf_error}"

    src_name = IT3_LANG_NAMES.get(from_lang, from_lang)
    tgt_name = IT3_LANG_NAMES.get(to_lang, to_lang)

    gloss_hint = ''
    if _glossary and from_lang == 'en' and to_lang == 'hi':
        pairs = ', '.join(f'"{k}"→"{v}"' for k, v in list(_glossary.items())[:30])
        gloss_hint = f'\n\nPreserve these technical terms exactly: {pairs}'

    prompt = (
        f"You are translating a technical user manual from {src_name} to {tgt_name}.\n"
        f"Rules:\n"
        f"1. Use formal imperative tone for instructions.\n"
        f"2. Keep WARNING, CAUTION, DANGER, NOTE labels unchanged and in uppercase.\n"
        f"3. Preserve all line breaks, blank lines, numbering, and indentation exactly.\n"
        f"4. Output only the translation — no explanations or commentary."
        f"{gloss_hint}\n\nText:\n{text}"
    )

    result = _it3_gguf_model.create_chat_completion(
        messages    = [{"role": "user", "content": prompt}],
        max_tokens  = 2048,
        temperature = 0.6,
        top_p       = 0.9,
    )
    return result['choices'][0]['message']['content'].strip()


# ── Unified entry point ──────────────────────────────────────────────────────

def _translate_with_it3(text, from_lang, to_lang):
    """Route to GGUF or transformers backend depending on GPU capability."""
    if _use_gguf_backend():
        return _translate_with_it3_gguf(text, from_lang, to_lang)

    # Transformers path (CC ≥ 7.5)
    global _it3_model, _it3_tokenizer
    import torch

    if _it3_model is None:
        if not _load_it3_model():
            return f"Error: could not load IT3 — {_it3_load_error}"

    src_name = IT3_LANG_NAMES.get(from_lang, from_lang)
    tgt_name = IT3_LANG_NAMES.get(to_lang, to_lang)

    gloss_hint = ''
    if _glossary and from_lang == 'en' and to_lang == 'hi':
        pairs = ', '.join(f'"{k}"→"{v}"' for k, v in list(_glossary.items())[:30])
        gloss_hint = f'\n\nPreserve these technical terms exactly as given: {pairs}'

    prompt = (
        f"You are translating a technical user manual from {src_name} to {tgt_name}.\n"
        f"Rules:\n"
        f"1. Use formal imperative tone for instructions (e.g. 'दबाएं', not 'दबाया जाना चाहिए').\n"
        f"2. Keep WARNING, CAUTION, DANGER, NOTE labels unchanged and in uppercase.\n"
        f"3. Preserve all line breaks, blank lines, numbering, and indentation exactly.\n"
        f"4. Translate technical terms consistently throughout the document.\n"
        f"5. Output only the translation — no explanations or commentary."
        f"{gloss_hint}\n\nText:\n{text}"
    )

    messages  = [{"role": "user", "content": prompt}]
    input_ids = _it3_tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    if input_ids.shape[-1] > 4096:
        input_ids = input_ids[:, -4096:]

    device    = next(_it3_model.parameters()).device
    input_ids = input_ids.to(device)

    with torch.no_grad():
        output = _it3_model.generate(
            input_ids,
            max_new_tokens = 2048,
            temperature    = 0.6,
            top_p          = 0.9,
            top_k          = 50,
            do_sample      = True,
        )

    new_tokens = output[0][input_ids.shape[-1]:]
    return _it3_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── Glossary helpers ────────────────────────────────────────────────────────

def _parse_xlsx_bytes(file_bytes):
    """Return {english: hindi} from an xlsx file's bytes."""
    zf = zipfile.ZipFile(io.BytesIO(file_bytes))
    ss_xml = zf.read('xl/sharedStrings.xml')
    ss_root = ET.fromstring(ss_xml)
    strings = []
    for si in ss_root.iter():
        if si.tag.endswith('}si'):
            t_parts = [e.text or '' for e in si.iter() if e.tag.endswith('}t') and e.text]
            strings.append(''.join(t_parts))
    sheet_xml = zf.read('xl/worksheets/sheet1.xml')
    sheet_root = ET.fromstring(sheet_xml)
    NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
    rows = []
    for row in sheet_root.iter(f'{NS}row'):
        vals = []
        for c in row:
            v = c.find(f'{NS}v')
            if v is not None and v.text is not None:
                vals.append(strings[int(v.text)] if c.get('t', '') == 's' else v.text)
            else:
                vals.append('')
        rows.append(vals)
    return _extract_pairs(rows)


def _parse_csv_bytes(file_bytes):
    """Return {english: hindi} from a CSV file's bytes."""
    import csv as _csv
    text = file_bytes.decode('utf-8-sig', errors='replace')
    rows = list(_csv.reader(io.StringIO(text)))
    return _extract_pairs(rows)


def _extract_pairs(rows):
    """Detect English/Hindi columns in a 2-D list and return {english: hindi}."""
    if not rows:
        return {}
    header = [c.strip().lower() for c in rows[0]]
    en_col = hi_col = None
    for i, h in enumerate(header):
        if any(kw in h for kw in ('english', 'en', 'term', 'word')) and en_col is None:
            en_col = i
        if any(kw in h for kw in ('hindi', 'hi', 'translation')) and hi_col is None:
            hi_col = i
    start = 1 if (en_col is not None and hi_col is not None) else 0
    if en_col is None or hi_col is None:
        # Fallback: skip numeric first column if present
        sample = next((r for r in rows if r), [])
        if len(sample) >= 3 and sample[0].strip().isdigit():
            en_col, hi_col = 1, 2
        else:
            en_col, hi_col = 0, 1
        start = 0
    result = {}
    skip = {'english', 'hindi', 'term', 'translation', 'en', 'hi', 'workshop hindi'}
    for row in rows[start:]:
        if len(row) <= max(en_col, hi_col):
            continue
        eng, hindi = row[en_col].strip(), row[hi_col].strip()
        if eng and hindi and eng.lower() not in skip:
            result[eng] = hindi
    return result


def _parse_glossary_file(file_bytes, filename):
    """Dispatch to xlsx or csv parser based on content / extension."""
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    if ext in ('xlsx', 'xls') or file_bytes[:2] == b'PK':
        return _parse_xlsx_bytes(file_bytes)
    return _parse_csv_bytes(file_bytes)


def _load_glossary():
    """Load the base English→Hindi glossary from 'hindi data.csv' (xlsx format)."""
    glossary = {}
    csv_path = Path(__file__).parent / "hindi data.csv"
    if not csv_path.exists():
        print("Warning: 'hindi data.csv' not found.")
        return glossary
    try:
        with open(str(csv_path), 'rb') as f:
            glossary = _parse_xlsx_bytes(f.read())
        print(f"Loaded {len(glossary)} base glossary terms from 'hindi data.csv'.")
    except Exception as e:
        print(f"Warning: Could not load glossary: {e}")
    return glossary


def _load_overrides():
    if OVERRIDES_PATH.exists():
        try:
            with open(OVERRIDES_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return {'added': data.get('added', {}), 'deleted': data.get('deleted', [])}
        except Exception:
            pass
    return {'added': {}, 'deleted': []}


def _save_overrides():
    with open(OVERRIDES_PATH, 'w', encoding='utf-8') as f:
        json.dump(_overrides, f, ensure_ascii=False, indent=2)


def _rebuild_glossary():
    global _glossary
    _glossary = dict(_base_glossary)
    _glossary.update(_overrides.get('learnt', {}))
    _glossary.update(_overrides.get('added', {}))
    for term in _overrides.get('deleted', []):
        _glossary.pop(term, None)


_DEVA_DIGITS = str.maketrans('0123456789', '०१२३४५६७८९')
_ARAB_DIGITS = str.maketrans('०१२३४५६७८९', '0123456789')
_PLACEHOLDER_BASE = 88800  # 5-digit range unlikely to appear in technical manuals


def _apply_glossary(text, glossary):
    """Replace known English terms with numeric placeholders before translation.

    Numbers are the only tokens IndicTrans2 reliably passes through unchanged.
    Uses 88800-88899 range; skips any numbers already present in the source.
    """
    restore_map = {}
    result = text
    counter = 0
    used = {int(n) for n in re.findall(r'\b\d{5}\b', text)}

    for term in sorted(glossary, key=len, reverse=True):
        pattern = r'(?i)(?<!\w)' + re.escape(term) + r'(?!\w)'
        if re.search(pattern, result):
            while (_PLACEHOLDER_BASE + counter) in used:
                counter += 1
            placeholder = str(_PLACEHOLDER_BASE + counter)
            result = re.sub(pattern, placeholder, result)
            restore_map[placeholder] = glossary[term]
            counter += 1
    return result, restore_map


def _restore_glossary(text, restore_map):
    """Swap numeric placeholders back to their Hindi equivalents.

    Handles both Arabic (88800) and Devanagari (८८८००) digit forms that the
    model may output.
    """
    for placeholder, hindi in restore_map.items():
        if placeholder in text:
            text = text.replace(placeholder, hindi)
        else:
            # Model may output Devanagari digits
            deva = placeholder.translate(_DEVA_DIGITS)
            text = text.replace(deva, hindi)
    return text


def hf_login():
    """Authenticate with HuggingFace. Uses HF_TOKEN env var first (Docker-friendly)."""
    try:
        from huggingface_hub import login, HfApi
        try:
            HfApi().whoami()
            print("Already logged in to HuggingFace.")
            return True
        except Exception:
            pass

        # Docker / non-interactive: read from environment variable
        token = os.environ.get('HF_TOKEN', '').strip()
        if token:
            print("Using HF_TOKEN from environment.")
        else:
            print("\n--- HuggingFace Login Required for IndicTrans2 ---")
            print("Tip: set HF_TOKEN env var to skip this prompt in Docker.")
            print("Get your token from: huggingface.co -> Settings -> Access Tokens")
            try:
                token = input("Enter your HuggingFace token: ").strip()
            except (EOFError, OSError):
                print("Non-interactive environment — set HF_TOKEN env var to authenticate.")
                return False

        if not token:
            print("No token provided. Skipping IndicTrans2 setup.")
            return False
        login(token=token)
        print("Logged in successfully.\n")
        return True
    except ImportError:
        print("huggingface_hub not found. Run: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"Login failed: {e}")
        return False


def initialize_translator():
    global indic_processor, _base_glossary, _overrides

    _base_glossary = _load_glossary()
    _overrides.update(_load_overrides())
    _rebuild_glossary()

    print("Initializing Argos Translate for Hindi...")
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()

    required_pairs = [("hi", "en"), ("en", "hi")]
    installed_packages = argostranslate.package.get_installed_packages()
    installed_pairs = [(pkg.from_code, pkg.to_code) for pkg in installed_packages]

    for from_code, to_code in required_pairs:
        if (from_code, to_code) in installed_pairs:
            print(f"{from_code}-{to_code} translation package already installed")
            continue
        package_to_install = next(
            (pkg for pkg in available_packages if pkg.from_code == from_code and pkg.to_code == to_code),
            None
        )
        if package_to_install:
            print(f"Downloading {from_code}-{to_code} translation package...")
            argostranslate.package.install_from_path(package_to_install.download())
            print(f"Package {from_code}-{to_code} installed successfully!")
        else:
            print(f"{from_code}-{to_code} package not found in available packages")

    if not TRANSFORMERS_AVAILABLE:
        print("\nTransformers or Torch not installed. Skipping IndicTrans2 Hindi setup.")
        print("To enable Hindi, run: pip install transformers sentencepiece torch")
        return

    if not hf_login():
        return

    print("Initializing IndicTrans2 for Hindi (this may take some time on first run)...")
    device = "cpu"
    if torch.cuda.is_available():
        cc = torch.cuda.get_device_capability(0)
        if cc >= (7, 5):
            device = "cuda"
        else:
            print(f"GPU compute capability {cc[0]}.{cc[1]} is below 7.5 — not supported by this PyTorch build. Falling back to CPU.")
    print(f"Using device: {device}")

    if INDIC_TOOLKIT_AVAILABLE:
        indic_processor = IndicProcessor(inference=True)
        print("IndicProcessor loaded.")
    else:
        print("Warning: IndicTransToolkit not installed — translation quality will be lower.")
        print("Install with: pip install IndicTransToolkit")

    pairs = {
        "en-hi": "ai4bharat/indictrans2-en-indic-1B",
        "hi-en": "ai4bharat/indictrans2-indic-en-1B",
    }

    for direction, model_name in pairs.items():
        print(f"Loading {model_name}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                dtype=torch.float16 if device == "cuda" else torch.float32,
            )
            model = model.to(device)
            model.eval()
            indic_tokenizers[direction] = tokenizer
            indic_models[direction] = model
            print(f"  {direction} model loaded.")
        except Exception as e:
            print(f"  Failed to load {direction} model: {e}")


_LEADING_SYMBOLS_RE = re.compile(r'^([^\w]*)', re.UNICODE)
_TRAILING_SYMBOLS_RE = re.compile(r'([^\w]*)$', re.UNICODE)


def _symbol_wrapper(text):
    """Split text into (leading_symbols, translatable_core, trailing_symbols).

    Unicode warning signs (⚠ △) sit outside \\w so they are captured as prefix/suffix
    and never sent to the translation model, preventing corruption or loss.
    """
    lm = _LEADING_SYMBOLS_RE.match(text)
    prefix = lm.group(1) if lm else ''
    rest = text[len(prefix):]
    tm = _TRAILING_SYMBOLS_RE.search(rest)
    suffix = tm.group(1) if tm else ''
    core = rest[: len(rest) - len(suffix)] if suffix else rest
    return prefix, core, suffix


# ── Hindi post-processing: known quality fixes for IndicTrans2 output ───────
_HINDI_POST_FIXES = [
    # "safe operating practices" → formal technical phrasing
    (re.compile(r'सुरक्षित\s+संचालन\s+प्रथाओं'), 'सुरक्षित संचालन प्रक्रियाओं'),
    (re.compile(r'संचालन\s+प्रथाओं'), 'संचालन प्रक्रियाओं'),
    # DANGER must express certainty ("will"), not mere possibility ("could")
    (re.compile(r'(DANGER[^\n]*)हो\s+सकत[ीि]\s+है'), r'\1निश्चित रूप से होगी'),
    (re.compile(r'(खतरा[^\n]*)हो\s+सकत[ीि]\s+है'), r'\1निश्चित रूप से होगी'),
    # Deduplicate the "संकेत का अर्थ संकेत" table-header artefact produced by model
    (re.compile(r'संकेत\s+का\s+अर्थ\s+संकेत'), 'संकेत का अर्थ'),
    # Formal connector in safety-related headings
    (re.compile(r'(चेतावन[ीि][^\n]*)\s+और\s+'), r'\1 एवं '),

    # ── User-manual imperative fixes ─────────────────────────────────────────
    # Passive "must be ensured" → imperative "ensure"
    (re.compile(r'सुनिश्चित\s+किया\s+जाना\s+चाहिए'), 'सुनिश्चित करें'),
    (re.compile(r'सुनिश्चित\s+करना\s+होगा'), 'सुनिश्चित करें'),
    # "must be checked" → "check"
    (re.compile(r'जाँच\s+की\s+जानी\s+चाहिए'), 'जाँच करें'),
    (re.compile(r'जांच\s+की\s+जानी\s+चाहिए'), 'जांच करें'),
    # NOTE label — model sometimes outputs verbose "टिप्पणी" instead of manual-standard "नोट"
    (re.compile(r'\bटिप्पणी\s*:'), 'नोट:'),
    # "refer to" verbose form → concise
    (re.compile(r'का\s+संदर्भ\s+लें'), 'देखें'),
    # Duplicate-word artefact: "चरण चरण" (step step)
    (re.compile(r'\bचरण\s+चरण\b'), 'चरण'),
    # "install किया जाना चाहिए" → "install करें" (mixed passive)
    (re.compile(r'(\w+)\s+किया\s+जाना\s+चाहिए'), r'\1 करें'),
]


def _postprocess_hindi(text):
    for pattern, repl in _HINDI_POST_FIXES:
        text = pattern.sub(repl, text)
    return text


_RE_ANGLE_TAG = re.compile(r'<([^>]{1,80})>')

# ── Phonetic English → Devanagari for loanword transliteration ───────────────
# Used when user tags <term>: saves transliteration, not translation.
# e.g. leg→लेग, motor→मोटर, brush→ब्रश, spring→स्प्रिंग

_EN_TRI = {   # 3-char clusters (checked first)
    'spr':'स्प्र','str':'स्ट्र','scr':'स्क्र','spl':'स्प्ल',
    'tch':'च','dge':'ज',
}
_EN_DI = {    # 2-char patterns (checked second)
    # Sound digraphs
    'sh':'श','ch':'च','th':'थ','ph':'फ','gh':'ग','ck':'क',
    'ng':'ंग','nk':'ंक','qu':'क्व','wh':'व','wr':'र',
    # C+r / C+l consonant clusters
    'br':'ब्र','cr':'क्र','dr':'ड्र','fr':'फ्र','gr':'ग्र','pr':'प्र','tr':'ट्र',
    'cl':'क्ल','fl':'फ्ल','gl':'ग्ल','pl':'प्ल','sl':'स्ल',
    # s-clusters
    'st':'स्ट','sp':'स्प','sk':'स्क','sc':'स्क','sn':'स्न','sm':'स्म','sw':'स्व',
    # Final-consonant clusters common in loanwords
    'mp':'म्प','nd':'न्ड','nt':'न्ट','lt':'ल्ट','lk':'ल्क','lm':'ल्म',
    'ft':'फ्ट','ct':'क्ट','pt':'प्ट','xt':'क्स्ट',
    # Vowel digraphs
    'ee':'ी','ea':'ी','oo':'ू',
    'ai':'ाइ','ay':'े','ey':'े','ie':'ी',
    'oi':'ॉइ','oy':'ॉइ','ou':'ाउ','ow':'ो',
    'ar':'ार','er':'र','ir':'र','ur':'र','or':'ोर',
}
_EN_SC = {    # single chars
    'a':'े','b':'ब','c':'क','d':'ड','e':'े','f':'फ',
    'g':'ग','h':'ह','i':'ि','j':'ज','k':'क','l':'ल',
    'm':'म','n':'न','o':'ो','p':'प','q':'क','r':'र',
    's':'स','t':'ट','u':'','v':'व','w':'व','x':'क्स',
    'y':'य','z':'ज़','-':'-',' ':' ',
}
_EN_FULL_V = {'a':'ए','e':'ए','i':'इ','o':'ओ','u':'उ'}


def _en_phonetic_to_deva(word):
    """Phonetic English → Devanagari (loanword style). e.g. leg→लेग, motor→मोटर."""
    out = []
    i = 0
    w = word.lower().strip()
    prev_vowel = True   # treat start-of-word as "after vowel" for full vowel forms

    while i < len(w):
        tri = w[i:i+3]
        if tri in _EN_TRI:
            out.append(_EN_TRI[tri])
            i += 3; prev_vowel = False; continue

        two = w[i:i+2]
        if two in _EN_DI:
            if two == 'or' and i + 2 >= len(w):
                out.append('र')            # final 'or' → just 'र'
            else:
                out.append(_EN_DI[two])
            i += 2; prev_vowel = two[-1] in 'aeiou'; continue

        c = w[i]
        if c == ' ':
            out.append(' '); prev_vowel = True
        elif c in 'aeiou':
            if prev_vowel:                 # word-start or after vowel → full form
                out.append(_EN_FULL_V.get(c, ''))
            else:
                out.append(_EN_SC.get(c, ''))
            prev_vowel = True
        else:
            out.append(_EN_SC.get(c, c))
            prev_vowel = False
        i += 1

    return ''.join(out)


def _auto_add_tagged_terms(core, engine):
    """Extract <term> tags from text, add new terms to glossary, return cleaned text."""
    matches = _RE_ANGLE_TAG.findall(core)
    if not matches:
        return core

    # Build a case-insensitive lookup set of existing glossary keys
    existing_lower = {k.lower() for k in _glossary}
    deleted_lower  = {t.lower() for t in _overrides.get('deleted', [])}

    changed = False
    for term in dict.fromkeys(matches):          # preserve order, deduplicate
        term = term.strip()
        if not term:
            continue
        if term.lower() in existing_lower:       # already in glossary (any casing)
            continue
        if term.lower() in deleted_lower:
            continue

        # Use phonetic transliteration — workshop terms must be Devanagari loanwords,
        # not semantic translations (leg→लेग not पैर, motor→मोटर not इंजन)
        hindi = _en_phonetic_to_deva(term)

        if not hindi:
            # Last resort: just use the term itself as a pass-through
            hindi = term

        _overrides.setdefault('learnt', {})[term] = hindi
        changed = True
        print(f"  Auto-glossary: <{term}> → {hindi}")

    if changed:
        _save_overrides()
        _rebuild_glossary()
    # Strip < > brackets regardless
    return _RE_ANGLE_TAG.sub(r'\1', core)


def translate_text(text, from_lang="hi", to_lang="en", engine="auto"):
    # ── Auto-add <term> tags BEFORE symbol_wrapper so <leg> isn't split as '<' + 'leg>' ──
    if from_lang == 'en' and to_lang == 'hi':
        text = _auto_add_tagged_terms(text, engine)

    prefix, core, suffix = _symbol_wrapper(text)
    if not core.strip():
        return text  # nothing translatable; return original (symbols intact)

    restore_map = {}
    # IT3 receives glossary terms inside its prompt; other engines use placeholders
    if from_lang == 'en' and to_lang == 'hi' and _glossary and engine != 'it3':
        core, restore_map = _apply_glossary(core, _glossary)

    result = _translate_core(core, from_lang, to_lang, engine)

    if restore_map:
        result = _restore_glossary(result, restore_map)

    if to_lang == 'hi':
        result = _postprocess_hindi(result)

    return prefix + result + suffix


def _translate_core(text, from_lang="hi", to_lang="en", engine="auto"):
    if engine == 'it3':
        return _translate_with_it3(text, from_lang, to_lang)

    direction = f"{from_lang}-{to_lang}"

    if "hi" in direction and engine != "argos":
        if not TRANSFORMERS_AVAILABLE or direction not in indic_models:
            return "Error: IndicTrans2 not loaded. Check startup logs."

        try:
            tokenizer = indic_tokenizers[direction]
            model = indic_models[direction]
            device = model.device

            src_lang = "eng_Latn" if from_lang == "en" else "hin_Deva"
            tgt_lang = "hin_Deva" if to_lang == "hi" else "eng_Latn"

            if INDIC_TOOLKIT_AVAILABLE and indic_processor:
                batch = indic_processor.preprocess_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)
                tokenizer.src_lang = src_lang
                enc = tokenizer(
                    batch,
                    truncation=True,
                    padding="longest",
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                # Pass only input_ids + attention_mask — extra None fields crash generate()
                model_inputs = {
                    "input_ids": enc["input_ids"].to(device),
                    "attention_mask": enc["attention_mask"].to(device),
                }
                with torch.no_grad():
                    generated_tokens = model.generate(
                        **model_inputs,
                        use_cache=False,
                        min_length=0,
                        max_length=512,
                        num_beams=4,
                        num_return_sequences=1,
                    )
                decoded = tokenizer.batch_decode(
                    generated_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                translations = indic_processor.postprocess_batch(decoded, lang=tgt_lang)
                return translations[0]
            else:
                return "Error: IndicTransToolkit is required. Run: pip install IndicTransToolkit"

        except Exception as e:
            return f"IndicTrans2 error: {str(e)}"

    try:
        installed_languages = argostranslate.translate.get_installed_languages()
        from_language = next((lang for lang in installed_languages if lang.code == from_lang), None)
        to_language = next((lang for lang in installed_languages if lang.code == to_lang), None)

        if not from_language or not to_language:
            return f"Error: Language pair {from_lang}-{to_lang} not installed."

        translation = from_language.get_translation(to_language)
        if not translation:
            return "Error: Translation not available"

        return translation.translate(text)
    except Exception as e:
        return f"Translation error: {str(e)}"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/it3/status')
def it3_status_route():
    return jsonify(_it3_status())


@app.route('/it3/load', methods=['POST'])
def it3_load_route():
    import threading
    if _use_gguf_backend():
        if _it3_gguf_model is not None:
            return jsonify({'status': 'already_loaded', 'backend': 'gguf'})
        if _it3_gguf_loading:
            return jsonify({'status': 'loading', 'backend': 'gguf'})
        threading.Thread(target=_load_it3_gguf, daemon=True).start()
        return jsonify({'status': 'started', 'backend': 'gguf'})
    else:
        if _it3_model is not None:
            return jsonify({'status': 'already_loaded', 'backend': 'transformers'})
        if _it3_loading:
            return jsonify({'status': 'loading', 'backend': 'transformers'})
        threading.Thread(target=_load_it3_model, daemon=True).start()
        return jsonify({'status': 'started', 'backend': 'transformers'})


@app.route('/glossary')
def glossary_info():
    added_keys  = set(_overrides.get('added',  {}).keys())
    learnt_keys = set(_overrides.get('learnt', {}).keys())
    def _src(k):
        if k in added_keys:  return 'custom'
        if k in learnt_keys: return 'learnt'
        return 'base'
    entries = [
        {'en': k, 'hi': v, 'source': _src(k)}
        for k, v in sorted(_glossary.items(), key=lambda x: x[0].lower())
    ]
    return jsonify({'terms': len(_glossary), 'entries': entries})


@app.route('/glossary/upload', methods=['POST'])
def glossary_upload():
    file = request.files.get('file')
    if not file or not file.filename:
        return jsonify({'error': 'No file provided'}), 400
    try:
        pairs = _parse_glossary_file(file.read(), file.filename)
        if not pairs:
            return jsonify({'error': 'No valid term pairs found in file'}), 400
        _overrides['added'].update(pairs)
        # Un-delete any terms that were just re-added
        _overrides['deleted'] = [d for d in _overrides['deleted'] if d not in pairs]
        _save_overrides()
        _rebuild_glossary()
        return jsonify({'added': len(pairs), 'terms': len(_glossary)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/glossary/term', methods=['POST'])
def glossary_add_term():
    data = request.get_json() or {}
    eng = data.get('en', '').strip()
    hindi = data.get('hi', '').strip()
    if not eng or not hindi:
        return jsonify({'error': 'Both English and Hindi terms are required'}), 400
    _overrides['added'][eng] = hindi
    if eng in _overrides['deleted']:
        _overrides['deleted'].remove(eng)
    _save_overrides()
    _rebuild_glossary()
    return jsonify({'ok': True, 'terms': len(_glossary)})


@app.route('/glossary/term', methods=['DELETE'])
def glossary_delete_term():
    data = request.get_json() or {}
    term = data.get('term', '').strip()
    if not term:
        return jsonify({'error': 'No term provided'}), 400
    _overrides.get('added',  {}).pop(term, None)
    _overrides.get('learnt', {}).pop(term, None)
    if term in _base_glossary and term not in _overrides.get('deleted', []):
        _overrides.setdefault('deleted', []).append(term)
    _save_overrides()
    _rebuild_glossary()
    return jsonify({'ok': True, 'terms': len(_glossary)})


IT3_MIN_CHARS = 300  # below this, IT3 on CPU is impractically slow — use IndicTrans2

@app.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.get_json()
        text = data.get('text', '')
        direction = data.get('direction', 'hi-en')
        engine = data.get('engine', 'auto')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        if '-' in direction:
            from_lang, to_lang = direction.split('-', 1)
        else:
            from_lang, to_lang = 'hi', 'en'

        # IT3 is a large LLM — full inference overhead even for one word.
        # Fall back to IndicTrans2 for short text so the UI stays responsive.
        fell_back = False
        if engine == 'it3' and len(text.strip()) < IT3_MIN_CHARS:
            engine = 'indic'
            fell_back = True

        result = translate_text(text, from_lang=from_lang, to_lang=to_lang, engine=engine)
        return jsonify({'translation': result, 'it3_fallback': fell_back})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/translate-stream', methods=['POST'])
def translate_stream():
    data = request.get_json()
    text = data.get('text', '')
    direction = data.get('direction', 'hi-en')
    engine = data.get('engine', 'auto')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    from_lang, to_lang = direction.split('-', 1) if '-' in direction else ('hi', 'en')

    def generate():
        lines = text.split('\n')
        # Index and text of non-empty lines only
        translatable = [(i, line) for i, line in enumerate(lines) if line.strip()]
        total = len(translatable)

        if total == 0:
            yield f"data: {json.dumps({'error': 'No text to translate'})}\n\n"
            return

        # Results array mirrors lines; empty lines stay empty
        results = [line if not line.strip() else '' for line in lines]

        yield f"data: {json.dumps({'total': total, 'status': 'start'})}\n\n"

        for idx, (line_idx, line) in enumerate(translatable):
            translated = translate_text(line, from_lang, to_lang, engine)
            results[line_idx] = translated
            current = '\n'.join(results)
            yield f"data: {json.dumps({'index': idx + 1, 'total': total, 'text': current})}\n\n"

        final_text = '\n'.join(results)
        yield f"data: {json.dumps({'complete': True, 'translation': final_text})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


def extract_pages(file, filename, use_ocr=False, ocr_lang='pol'):
    """Return list of page strings. PDF = real pages; DOCX/TXT = ~10 paragraphs per page."""
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''

    if ext == 'pdf':
        pdf_bytes = file.read() if hasattr(file, 'read') else bytes(file)
        pages = []

        if not use_ocr:
            if not PYPDF_AVAILABLE:
                raise ValueError("pypdf not installed. Run: pip install pypdf")
            reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
            pages = [p.extract_text() or '' for p in reader.pages]

        # Apply OCR: either forced, or as automatic fallback for image-only pages
        if PYMUPDF_AVAILABLE and PYTESSERACT_AVAILABLE:
            if use_ocr or any(not p.strip() for p in pages):
                pdf_doc = fitz.open(stream=pdf_bytes, filetype='pdf')
                if use_ocr:
                    pages = []
                    for page_obj in pdf_doc:
                        mat = fitz.Matrix(2, 2)
                        pix = page_obj.get_pixmap(matrix=mat, alpha=False)
                        img = PilImage.open(io.BytesIO(pix.tobytes('png')))
                        pages.append(pytesseract.image_to_string(img, lang=ocr_lang))
                else:
                    for i in range(len(pages)):
                        if not pages[i].strip() and i < len(pdf_doc):
                            mat = fitz.Matrix(2, 2)
                            pix = pdf_doc[i].get_pixmap(matrix=mat, alpha=False)
                            img = PilImage.open(io.BytesIO(pix.tobytes('png')))
                            pages[i] = pytesseract.image_to_string(img, lang=ocr_lang)
                pdf_doc.close()
        elif use_ocr:
            missing = []
            if not PYMUPDF_AVAILABLE:
                missing.append("pymupdf")
            if not PYTESSERACT_AVAILABLE:
                missing.append("pytesseract pillow")
            raise ValueError(
                f"OCR requires: pip install {' '.join(missing)}. "
                "Also install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki"
            )

        return [p for p in pages if p.strip()]

    elif ext in ('docx', 'doc'):
        if not DOCX_AVAILABLE:
            raise ValueError("python-docx not installed. Run: pip install python-docx")
        doc = DocxDocument(file)
        all_paras = [p.text for p in doc.paragraphs]
        pages, bucket, count = [], [], 0
        for p in all_paras:
            bucket.append(p)
            if p.strip():
                count += 1
            if count >= 10:
                pages.append('\n'.join(bucket))
                bucket, count = [], 0
        if bucket:
            pages.append('\n'.join(bucket))
        return [p for p in pages if p.strip()]

    elif ext == 'txt':
        lines = file.read().decode('utf-8', errors='replace').split('\n')
        pages, bucket, words = [], [], 0
        for line in lines:
            bucket.append(line)
            words += len(line.split())
            if words >= 300:
                pages.append('\n'.join(bucket))
                bucket, words = [], 0
        if bucket:
            pages.append('\n'.join(bucket))
        return [p for p in pages if p.strip()]

    else:
        raise ValueError(f"Unsupported file type: .{ext}. Supported: TXT, PDF, DOCX")


_LANG_TO_TESSDATA = {'en': 'eng', 'hi': 'hin'}

WNS = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
# Fonts that render symbols/dingbats — their text should not be translated
_SYMBOL_FONTS = {'symbol', 'wingdings', 'wingdings 2', 'wingdings 3', 'webdings', 'marlett'}


def _run_is_translatable(r):
    """True if run contains plain text that should be translated (not image, not symbol font)."""
    if r.find(f'{WNS}drawing') is not None or r.find(f'{WNS}pict') is not None:
        return False
    if r.find(f'{WNS}t') is None:
        return False
    rPr = r.find(f'{WNS}rPr')
    if rPr is not None:
        rFonts = rPr.find(f'{WNS}rFonts')
        if rFonts is not None:
            font = (rFonts.get(f'{WNS}ascii') or rFonts.get(f'{WNS}hAnsi') or '').lower()
            if font in _SYMBOL_FONTS:
                return False
    return True


def _para_source_text(para):
    """Extract only translatable text from a paragraph (skip images and symbol runs)."""
    parts = []
    for r in para._p.findall(f'{WNS}r'):
        if _run_is_translatable(r):
            t = r.find(f'{WNS}t')
            if t is not None and t.text:
                parts.append(t.text)
    return ''.join(parts).strip()


def _apply_translation(para, translated):
    """Replace translatable text runs with translated text, leaving images/symbols intact."""
    from docx.oxml import OxmlElement
    p = para._p
    text_runs = [r for r in p.findall(f'{WNS}r') if _run_is_translatable(r)]
    if not text_runs:
        return

    # Use rPr from the run with the most text so that a short bold label
    # ("Manuals and Signs:") doesn't make the entire translated paragraph bold.
    def _run_text_len(r):
        t = r.find(f'{WNS}t')
        return len(t.text) if t is not None and t.text else 0

    dominant_run = max(text_runs, key=_run_text_len)

    new_r = OxmlElement('w:r')
    dom_rPr = dominant_run.find(f'{WNS}rPr')
    if dom_rPr is not None:
        new_r.append(copy.deepcopy(dom_rPr))

    new_t = OxmlElement('w:t')
    new_t.text = translated
    new_t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    new_r.append(new_t)

    text_runs[0].addprevious(new_r)
    for r in text_runs:
        p.remove(r)



def _heading_level(para):
    """Return 1-9 if para has a Heading style, else 0."""
    name = (para.style.name or '').lower()
    if 'heading' not in name:
        return 0
    for n in range(1, 10):
        if str(n) in name:
            return n
    return 1


def _collect_table_paras(table):
    """Collect translatable paragraphs from a table, deduplicating merged cells and nesting."""
    paras = []
    seen = set()
    for row in table.rows:
        for cell in row.cells:
            tc_id = id(cell._tc)
            if tc_id in seen:
                continue
            seen.add(tc_id)
            for p in cell.paragraphs:
                if _para_source_text(p):
                    paras.append(p)
            for nested in cell.tables:
                paras.extend(_collect_table_paras(nested))
    return paras


def _docx_sse(file_bytes, from_lang, to_lang, engine):
    """SSE generator: translates DOCX in-place, streaming one chunk per paragraph."""
    try:
        doc = DocxDocument(io.BytesIO(file_bytes))

        all_paras = [p for p in doc.paragraphs if _para_source_text(p)]
        for table in doc.tables:
            all_paras.extend(_collect_table_paras(table))

        total = len(all_paras)
        if total == 0:
            yield f"data: {json.dumps({'error': 'No text found in document'})}\n\n"
            return

        yield f"data: {json.dumps({'total': total, 'status': 'start'})}\n\n"

        current_section = ''  # translated heading of the current section

        for i, para in enumerate(all_paras):
            src = _para_source_text(para)
            if not src:
                continue

            if i % 10 == 0:
                yield ": keep-alive\n\n"

            try:
                is_heading = _heading_level(para) > 0

                if engine == 'it3' and not is_heading and current_section:
                    # Prepend section heading so IT3 stays contextually consistent
                    context_src = f"[Section: {current_section}]\n{src}"
                    translated = translate_text(context_src, from_lang, to_lang, engine)
                    # Strip the echoed context line if the model included it
                    lines = translated.split('\n')
                    if lines and lines[0].startswith('[') and ']' in lines[0]:
                        translated = '\n'.join(lines[1:]).strip()
                else:
                    translated = translate_text(src, from_lang, to_lang, engine)

                if is_heading:
                    current_section = translated

                _apply_translation(para, translated)
            except Exception as para_err:
                translated = f"[ERROR: {para_err}]"
                print(f"  Para {i+1} failed: {para_err}")

            progress = int((i + 1) / total * 100)
            yield f"data: {json.dumps({'chunk': {'src': src, 'translated': translated}, 'index': i + 1, 'total': total, 'progress': progress})}\n\n"

        buf = io.BytesIO()
        doc.save(buf)
        buf.seek(0)
        doc_b64 = base64.b64encode(buf.read()).decode('utf-8')
        yield f"data: {json.dumps({'complete': True, 'docx_b64': doc_b64})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


def _plain_sse(file_bytes, filename, from_lang, to_lang, engine, use_ocr=False):
    """SSE generator for PDF / TXT: streams one chunk per non-empty line."""
    try:
        ocr_lang = _LANG_TO_TESSDATA.get(from_lang, 'eng')
        pages = extract_pages(io.BytesIO(file_bytes), filename, use_ocr=use_ocr, ocr_lang=ocr_lang)
        if not pages:
            yield f"data: {json.dumps({'error': 'No text found in document'})}\n\n"
            return

        all_lines = [
            line.strip()
            for page_text in pages
            for line in page_text.split('\n')
            if line.strip()
        ]
        total = len(all_lines)
        yield f"data: {json.dumps({'total': total, 'status': 'start'})}\n\n"

        for i, line in enumerate(all_lines):
            # Keep-alive comment every 10 lines so the SSE connection doesn't time out
            if i % 10 == 0:
                yield ": keep-alive\n\n"

            try:
                translated = translate_text(line, from_lang=from_lang, to_lang=to_lang, engine=engine)
            except Exception as line_err:
                # One bad line should not stop the whole document — skip it
                translated = f"[ERROR: {line_err}]"
                print(f"  Line {i+1} failed: {line_err}")

            progress = int((i + 1) / total * 100)
            yield f"data: {json.dumps({'chunk': {'src': line, 'translated': translated}, 'index': i + 1, 'total': total, 'progress': progress})}\n\n"

        yield f"data: {json.dumps({'complete': True})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


@app.route('/translate-doc', methods=['POST'])
def translate_doc():
    file = request.files.get('file')
    if not file or not file.filename:
        return jsonify({'error': 'No file provided'}), 400

    file_bytes = file.read()
    filename = file.filename
    direction = request.form.get('direction', 'hi-en')
    engine = request.form.get('engine', 'auto')
    use_ocr = request.form.get('use_ocr', '').lower() in ('true', '1', 'yes')
    from_lang, to_lang = direction.split('-', 1) if '-' in direction else ('hi', 'en')
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''

    # GGUF backend works on any GPU (CC 3.5+) and CPU — no fallback needed.
    # Only fall back to IndicTrans2 if IT3 has no GGUF file AND no CUDA GPU.
    if engine == 'it3':
        import torch
        no_cuda   = not torch.cuda.is_available()
        no_gguf   = _find_local_gguf() is None
        if no_cuda and no_gguf and _use_gguf_backend():
            engine = 'indic'
            def _warn_then(gen):
                yield f"data: {json.dumps({'warning': 'No GGUF file found and no GPU — falling back to IndicTrans2. Place a .gguf file in models/it3_gguf/ to enable IT3.'})}\n\n"
                yield from gen
            gen_inner = _docx_sse(file_bytes, from_lang, to_lang, engine) \
                if ext in ('docx', 'doc') and DOCX_AVAILABLE \
                else _plain_sse(file_bytes, filename, from_lang, to_lang, engine, use_ocr=use_ocr)
            gen = _warn_then(gen_inner)
        else:
            gen = _docx_sse(file_bytes, from_lang, to_lang, engine) \
                if ext in ('docx', 'doc') and DOCX_AVAILABLE \
                else _plain_sse(file_bytes, filename, from_lang, to_lang, engine, use_ocr=use_ocr)
    else:
        gen = _docx_sse(file_bytes, from_lang, to_lang, engine) \
            if ext in ('docx', 'doc') and DOCX_AVAILABLE \
            else _plain_sse(file_bytes, filename, from_lang, to_lang, engine, use_ocr=use_ocr)

    return Response(
        stream_with_context(gen),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@app.route('/build-docx', methods=['POST'])
def build_docx():
    if not DOCX_AVAILABLE:
        return jsonify({'error': 'python-docx not installed'}), 500
    data = request.get_json()
    text = data.get('text', '')
    filename = data.get('filename', 'translation')
    doc = DocxDocument()
    for para in text.split('\n'):
        doc.add_paragraph(para)
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return send_file(
        buf,
        mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        as_attachment=True,
        download_name=f"{filename}_translated.docx",
    )


if __name__ == '__main__':
    print("Initializing translator...")
    initialize_translator()
    print("\nStarting Flask server...")
    print("Open your browser and go to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
