# Docker Setup & Transfer Guide

---

## WINDOWS SETUP (Full Steps)

### Step W1 — Enable WSL2

Open PowerShell as Administrator and run:
```powershell
wsl --install
```
Restart your PC when prompted.

If WSL is already installed, make sure it is version 2:
```powershell
wsl --set-default-version 2
```

Verify:
```powershell
wsl --status
```
Should say "Default Version: 2".

---

### Step W2 — Install NVIDIA Driver

1. Open Device Manager → Display Adapters → note your GPU name
2. Download the latest driver from https://www.nvidia.com/Download/index.aspx
3. Install and restart
4. Open PowerShell and verify:
```powershell
nvidia-smi
```
You should see your GPU and CUDA version listed.

---

### Step W3 — Install Docker Desktop

1. Download from https://www.docker.com/products/docker-desktop
2. During install, make sure "Use WSL 2 instead of Hyper-V" is checked
3. Restart your PC
4. Launch Docker Desktop and wait for it to say "Engine running" (green icon in taskbar)
5. Open Docker Desktop → Settings → General → confirm "Use the WSL 2 based engine" is ON

Verify Docker works:
```powershell
docker --version
docker compose version
```

---

### Step W4 — Enable GPU support in Docker Desktop

1. Open Docker Desktop
2. Go to Settings → Resources → WSL Integration
3. Turn on integration for your default WSL distro
4. Click "Apply & Restart"

GPU access is automatic on Windows with Docker Desktop when your NVIDIA driver is installed — no extra toolkit needed.

Verify GPU works inside Docker:
```powershell
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```
You should see your GPU listed inside the container.

---

### Step W5 — Get your HuggingFace token

1. Go to https://huggingface.co/settings/tokens
2. Click "New token" → Name it anything → Role: Read
3. Copy the token (starts with `hf_...`)

---

### Step W6 — Transfer the project to the new machine

**Option A — USB / network copy (simplest)**

Copy the entire `translator\` folder to the new machine.
Make sure these files are included:
```
translator\
├── Dockerfile
├── docker-compose.yml
├── P2E.py
├── templates\index.html
├── hindi_data.csv
├── glossary_overrides.json
└── DOCKER_SETUP.md
```

**Option B — Save Docker image to file (for machines with no internet)**

Run these on your CURRENT machine after the first build:
```powershell
# Build the image first
docker compose build

# Save to a file (this will be ~8-10 GB)
docker save translator-translator -o translator_image.tar
```
Copy `translator_image.tar` and the project folder to the new machine via USB, then on the new machine:
```powershell
docker load -i translator_image.tar
```

---

### Step W7 — Create the .env file

Inside the `translator\` folder, create a new text file named `.env` (no extension).

Open Notepad and paste:
```
HF_TOKEN=hf_your_token_here
```
Replace `hf_your_token_here` with your actual token from Step W5.

Save as `All Files` type and name it exactly `.env` (not `.env.txt`).

To check it was saved correctly in PowerShell:
```powershell
Get-Content .env
```
Should print your token line.

---

### Step W8 — Build and run

Open PowerShell inside the `translator\` folder:
```powershell
cd C:\path\to\translator
```

First time (builds image + downloads models, ~10-15 GB, takes 20-40 minutes):
```powershell
docker compose up --build
```

You will see model download progress in the terminal. When you see:
```
* Running on http://0.0.0.0:5000
```
Open your browser and go to: http://localhost:5000

---

### Step W9 — Stop and restart

```powershell
# Stop (keep models cached)
docker compose down

# Start again next time (fast, no re-download)
docker compose up

# Rebuild only if you changed code
docker compose up --build
```

---

### Step W10 — Run in background (optional)

To run without keeping the terminal open:
```powershell
docker compose up -d
```
Stop it later with:
```powershell
docker compose down
```

---

## LINUX SETUP (Ubuntu/Debian)

### Step L1 — Install Docker
```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker
```

### Step L2 — Install NVIDIA Container Toolkit
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify:
```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### Step L3 — Run the app
```bash
cd /path/to/translator
echo "HF_TOKEN=hf_your_token_here" > .env
docker compose up --build
```

---

## Troubleshooting

**"docker: command not found"**
- Windows: Docker Desktop is not running — open it from the Start menu and wait for the green icon
- Linux: Run `sudo apt-get install docker.io` or follow Step L1 again

**GPU not detected inside container**
- Windows: Go to Docker Desktop → Settings → Resources → WSL Integration → enable and Apply & Restart
- Linux: Run `sudo systemctl restart docker` then test with nvidia-smi inside container
- Make sure your NVIDIA driver is installed: run `nvidia-smi` outside Docker first

**".env file not found" or token not working**
- Make sure the file is named `.env` not `.env.txt`
- In PowerShell: `Get-Content .env` should show `HF_TOKEN=hf_...`
- No spaces around the `=` sign

**Port 5000 already in use**
- Edit `docker-compose.yml`, change `"5000:5000"` to `"5001:5000"`
- Then open http://localhost:5001

**Models re-downloading every time**
- Do NOT run `docker compose down -v` — the `-v` flag deletes the model cache volumes
- Just use `docker compose down` to stop

**Out of VRAM during IT3**
- The app auto-switches to 4-bit quantization for GPUs under 12 GB
- Make sure `bitsandbytes` is installed (already in the Dockerfile)

**Build fails on "IndicTransToolkit"**
- This package may need to be installed from GitHub. If the build fails on this line, edit Dockerfile and replace:
  ```
  IndicTransToolkit \
  ```
  with:
  ```
  git+https://github.com/AI4Bharat/IndicTransToolkit.git \
  ```
  Then add `git` to the apt-get install line.

---

## File structure on new machine

```
translator\
├── .env                     ← YOU CREATE THIS (your HuggingFace token)
├── .dockerignore
├── docker-compose.yml
├── Dockerfile
├── P2E.py
├── templates\
│   └── index.html
├── hindi_data.csv
├── glossary_overrides.json  ← glossary edits persist via Docker volume
└── DOCKER_SETUP.md          ← this file
```

Docker volumes (stored automatically by Docker, not in the folder):
- `hf_cache` — all HuggingFace models (IndicTrans2, IT3)
- `argos_models` — Argos translation packages
