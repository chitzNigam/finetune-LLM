#!/bin/bash
# scripts/setup_wsl.sh
# One-shot WSL2 environment setup for whatsapp-style-model
# Run: chmod +x scripts/setup_wsl.sh && ./scripts/setup_wsl.sh

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║    WhatsApp Style Model — WSL2 Setup     ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo -e "${YELLOW}[INFO] This script downloads code from external sources (Miniconda, PyTorch wheels, Ollama).${NC}"
echo -e "${YELLOW}[INFO] Review scripts/setup_wsl.sh before running it on a machine that contains private chat data.${NC}"

# ── 1. Check WSL ──────────────────────────────────────────────────────────────
if ! grep -qi microsoft /proc/version 2>/dev/null; then
    echo -e "${YELLOW}[WARN] Not running in WSL. Continuing anyway...${NC}"
fi

# ── 2. System packages ────────────────────────────────────────────────────────
echo -e "${GREEN}[1/7] Installing system packages...${NC}"
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3.11 python3.11-venv python3-pip \
    git curl wget build-essential \
    libssl-dev libffi-dev \
    unzip

# ── 3. Check NVIDIA ───────────────────────────────────────────────────────────
echo -e "${GREEN}[2/7] Checking NVIDIA GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo -e "${GREEN}  ✓ GPU detected${NC}"
else
    echo -e "${RED}  ✗ nvidia-smi not found${NC}"
    echo "    Install latest NVIDIA drivers on Windows (≥ 525.x)"
    echo "    Then restart WSL: wsl --shutdown from PowerShell"
fi

# ── 4. Conda ──────────────────────────────────────────────────────────────────
echo -e "${GREEN}[3/7] Setting up Miniconda...${NC}"
if ! command -v conda &> /dev/null; then
    echo -e "${YELLOW}  → Downloading Miniconda installer from repo.anaconda.com${NC}"
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
         -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda"
    echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
    export PATH="$HOME/miniconda/bin:$PATH"
    conda init bash
    echo -e "${GREEN}  ✓ Miniconda installed${NC}"
else
    echo -e "${GREEN}  ✓ Conda already installed${NC}"
fi

source "$HOME/miniconda/etc/profile.d/conda.sh" 2>/dev/null || true

# ── 5. Conda environment ──────────────────────────────────────────────────────
echo -e "${GREEN}[4/7] Creating conda environment 'whatsapp-model'...${NC}"
conda create -n whatsapp-model python=3.11 -y -q
conda activate whatsapp-model || source activate whatsapp-model

# ── 6. PyTorch ────────────────────────────────────────────────────────────────
echo -e "${GREEN}[5/7] Installing PyTorch with CUDA 12.1...${NC}"
pip install -q torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# ── 7. Python dependencies ────────────────────────────────────────────────────
echo -e "${GREEN}[6/7] Installing Python dependencies...${NC}"
pip install -q -r requirements.txt

# ── 8. Ollama ─────────────────────────────────────────────────────────────────
echo -e "${GREEN}[7/7] Installing Ollama...${NC}"
if ! command -v ollama &> /dev/null; then
    echo -e "${YELLOW}  → Running Ollama install script from ollama.com${NC}"
    curl -fsSL https://ollama.com/install.sh | sh
    echo -e "${GREEN}  ✓ Ollama installed${NC}"
else
    echo -e "${GREEN}  ✓ Ollama already installed${NC}"
fi

# ── 9. Project dirs ───────────────────────────────────────────────────────────
echo ""
echo "Setting up project directories..."
mkdir -p data/{raw,processed,exports}
mkdir -p models/{checkpoints,final}

# Copy example contacts if not exists
if [ ! -f data/contacts.json ]; then
    cp data/contacts.example.json data/contacts.json 2>/dev/null || true
    echo -e "${YELLOW}  → Edit data/contacts.json with your contact mappings${NC}"
fi

# ── 10. Verify ────────────────────────────────────────────────────────────────
echo ""
echo "Running verification..."
python scripts/verify_setup.py

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║              Setup Complete!             ║"
echo "╠══════════════════════════════════════════╣"
echo "║  Next steps:                             ║"
echo "║  1. conda activate whatsapp-model        ║"
echo "║  2. Edit data/contacts.json              ║"
echo "║  3. Drop .txt exports in data/raw/       ║"
echo "║  4. python scripts/run_pipeline.py       ║"
echo "║  5. python src/training/train.py         ║"
echo "╚══════════════════════════════════════════╝"
echo ""
