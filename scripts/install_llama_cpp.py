"""Install llama-cpp-python with the best backend for this machine.

Order of preference:
  1. NVIDIA GPU + CUDA wheel available for this OS      -> real GPU offload
  2. CPU wheel known to work on this machine's CPU       -> safe fallback

Windows has no prebuilt wheel that combines CUDA support with the gemma3
architecture (needed for IndicTrans3 GGUF) -- see README note below -- so
Windows always installs the pinned CPU wheel. Linux/macOS get a real
CUDA/Metal attempt first, with a CPU fallback if it fails.

Every install is smoke-tested with a subprocess `import llama_cpp` before
being accepted, so a wheel that installs but crashes on this CPU's
instruction set (e.g. AVX-512-only builds on older Xeons) is rejected and
the next candidate is tried.
"""
import platform
import re
import subprocess
import sys

# Newest version confirmed to support the gemma3 architecture (IndicTrans3 GGUF).
# Newer CPU wheels (0.3.20+) use instructions some older CPUs (pre-AVX512 Xeons)
# don't have, so we pin here rather than floating to "latest".
PINNED_VERSION = "0.3.19"

# Newest version with a Windows CUDA wheel. CUDA builds beyond this dropped
# Windows entirely (Linux-only) on abetlen's index.
WINDOWS_CUDA_VERSION = "0.3.4"

CUDA_INDEXES = ["cu132", "cu130", "cu125", "cu124", "cu123", "cu122", "cu121", "cu118"]
INDEX_BASE = "https://abetlen.github.io/llama-cpp-python/whl"


def _run(cmd, **kw):
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, **kw)


def detect_nvidia_cuda_version():
    """Return (major, minor) of the CUDA version reported by the driver, or None."""
    try:
        out = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out.stdout)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def pick_cuda_index(detected):
    """Pick the highest available cuXXX index that the driver supports."""
    major, minor = detected
    driver_num = major * 100 + minor * 10 if minor < 10 else major * 100 + minor
    # Index tags encode as cuMAJOR MINOR, e.g. cu121 = 12.1, cu118 = 11.8
    best = None
    for tag in CUDA_INDEXES:
        digits = tag[2:]
        t_major = int(digits[:2])
        t_minor = int(digits[2:])
        tag_num = t_major * 100 + t_minor
        drv_num = major * 100 + minor
        if tag_num <= drv_num:
            best = tag
            break
    return best


def smoke_test(python_exe):
    result = _run(
        [python_exe, "-c", "from llama_cpp import Llama; print('OK')"],
        capture_output=True, text=True,
    )
    return result.returncode == 0 and "OK" in result.stdout


def pip_install(python_exe, version, index_tag):
    url = f"{INDEX_BASE}/{index_tag}"
    result = _run([
        python_exe, "-m", "pip", "install",
        f"llama-cpp-python=={version}",
        "--extra-index-url", url,
        "--force-reinstall", "--no-cache-dir",
    ])
    return result.returncode == 0


def pip_install_default(python_exe, version):
    result = _run([python_exe, "-m", "pip", "install", f"llama-cpp-python=={version}",
                    "--force-reinstall", "--no-cache-dir"])
    return result.returncode == 0


def main():
    python_exe = sys.executable
    system = platform.system()
    print(f"Detected OS: {system}")

    candidates = []  # list of (label, install_fn)

    if system == "Windows":
        cuda = detect_nvidia_cuda_version()
        if cuda:
            print(f"NVIDIA GPU detected (driver CUDA {cuda[0]}.{cuda[1]}), "
                  f"but no Windows wheel combines CUDA + gemma3 support "
                  f"(newest Windows CUDA wheel is {WINDOWS_CUDA_VERSION}, "
                  f"gemma3 needs {PINNED_VERSION}+). Using CPU wheel.")
        candidates.append(("CPU (pinned)", lambda: pip_install(python_exe, PINNED_VERSION, "cpu")))

    elif system == "Linux":
        cuda = detect_nvidia_cuda_version()
        if cuda:
            tag = pick_cuda_index(cuda)
            if tag:
                print(f"NVIDIA GPU detected (driver CUDA {cuda[0]}.{cuda[1]}) -> trying {tag}")
                candidates.append((f"CUDA {tag}", lambda t=tag: pip_install(python_exe, PINNED_VERSION, t)))
        candidates.append(("CPU (pinned)", lambda: pip_install(python_exe, PINNED_VERSION, "cpu")))

    elif system == "Darwin":
        candidates.append(("Metal/default", lambda: pip_install_default(python_exe, PINNED_VERSION)))
        candidates.append(("CPU (pinned)", lambda: pip_install(python_exe, PINNED_VERSION, "cpu")))

    else:
        candidates.append(("CPU (pinned)", lambda: pip_install(python_exe, PINNED_VERSION, "cpu")))

    for label, install_fn in candidates:
        print(f"\n--- Attempting: {label} ---")
        if not install_fn():
            print(f"  install failed, trying next option")
            continue
        if smoke_test(python_exe):
            print(f"  {label}: import OK - using this build")
            return 0
        print(f"  {label}: installed but failed smoke test (likely unsupported "
              f"CPU/GPU instructions) - trying next option")

    print("\nERROR: no llama-cpp-python build could be installed and imported "
          "successfully on this machine.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
