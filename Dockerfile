FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-dev \
    tesseract-ocr tesseract-ocr-hin tesseract-ocr-eng \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

WORKDIR /app

# Install PyTorch with CUDA 12.4 first (large, cache this layer)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install all other dependencies
RUN pip install --no-cache-dir \
    Flask==3.0.0 \
    argostranslate==1.9.1 \
    transformers \
    sentencepiece \
    sacremoses \
    accelerate \
    bitsandbytes \
    IndicTransToolkit \
    llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 \
    pypdf \
    pymupdf \
    pytesseract \
    Pillow \
    python-docx \
    huggingface_hub \
    openpyxl

# Copy app files (models/hf_cache come from volumes, not the image)
COPY P2E.py .
COPY templates/ templates/
COPY "hindi data.odt" .
COPY hindi_data.csv .
COPY glossary_overrides.json .

EXPOSE 5000

CMD ["python", "P2E.py"]
