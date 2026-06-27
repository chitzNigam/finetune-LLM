# WSL2 Setup Guide

Complete setup for WSL2 + CUDA + Python environment for training.

---

## 1. Enable WSL2

Open **PowerShell as Administrator** on Windows:

```powershell
wsl --install
wsl --set-default-version 2
# Restart your machine
```

Install Ubuntu 22.04 from Microsoft Store, then open it.

---

## 2. Verify CUDA Access in WSL2

NVIDIA drivers are installed on **Windows**, not WSL. WSL2 passes through CUDA automatically.

```bash
# Inside WSL2
nvidia-smi
# Should show your 4070 Ti Super and CUDA version
```

If `nvidia-smi` fails:
- Update your Windows NVIDIA driver to latest (≥ 525.x)
- Ensure WSL2 kernel is up to date: `wsl --update` from PowerShell

---

## 3. Install System Dependencies

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
    python3.11 python3.11-venv python3-pip \
    git curl wget build-essential \
    libssl-dev libffi-dev \
    ffmpeg libsm6 libxext6
```

---

## 4. Install Miniconda (Recommended over raw venv)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
conda init bash
source ~/.bashrc
```

---

## 5. Create Project Environment

```bash
conda create -n whatsapp-model python=3.11 -y
conda activate whatsapp-model
```

---

## 6. Install PyTorch with CUDA

```bash
# For CUDA 12.1 (check your version with nvidia-smi)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# Should print: True, NVIDIA GeForce RTX 4070 Ti SUPER
```

If `nvidia-smi` shows a newer driver/CUDA runtime such as 13.x, that is still fine.
The Windows driver is backward-compatible with the `cu121` PyTorch wheels used here.

---

## 7. Install Project Dependencies

```bash
cd whatsapp-style-model
pip install -r requirements.txt
```

---

## 8. Install Ollama (for inference)

```bash
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve &

# Pull base model (do this before training, useful for testing)
ollama pull qwen2.5:7b
```

---

## 9. Configure WSL2 Memory (Important)

By default WSL2 can grab too much or too little RAM. Create/edit `C:\Users\<YourName>\.wslconfig` on **Windows**:

```ini
[wsl2]
memory=24GB          # Leave some for Windows
processors=8
swap=8GB
gpuSupport=true
```

Restart WSL: `wsl --shutdown` from PowerShell, then reopen Ubuntu.

---

## 10. Verify Full Setup

```bash
python scripts/verify_setup.py
```

Expected output:
```
✅ Python 3.11
✅ CUDA available — RTX 4070 Ti SUPER (16376 MB)
✅ PyTorch 2.x
✅ transformers installed
✅ peft installed
✅ bitsandbytes installed
✅ trl installed
✅ Ollama running
```

---

## Common WSL2 Issues

**`nvidia-smi` not found**
→ Install latest NVIDIA Game Ready or Studio driver on Windows (not in WSL)

**CUDA out of memory during training**
→ Reduce `per_device_train_batch_size` to 1 in `src/training/config.py`

**Slow file I/O**
→ Keep your project files inside WSL filesystem (`~/`) not on `/mnt/c/`. Windows ↔ WSL file access is slow.

**`bitsandbytes` CUDA error**
```bash
pip install bitsandbytes --upgrade
# If still failing:
pip install bitsandbytes==0.43.3
```

**Merge step runs out of VRAM**
→ Close other GPU-heavy apps first. `src/training/merge.py` loads the full base model in FP16, so merge can be tighter on 16 GB than 4-bit training or inference.
