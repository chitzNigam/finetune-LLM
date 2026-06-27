# Training Guide

Everything you need to know about the fine-tuning process.

---

## Overview

We use **QLoRA** (Quantized Low-Rank Adaptation) to fine-tune `Qwen2.5-7B-Instruct` on your WhatsApp data.

- Base model is loaded in **4-bit NF4** quantization (~6GB VRAM)
- Only **LoRA adapter weights** are trained (~300MB)
- Total VRAM during training: ~11–13GB on your 4070 Ti Super
- Estimated training time: **3–6 hours** for 3 epochs on 10k samples

---

## Before You Train

```bash
# Ensure pipeline has been run
ls data/exports/*.jsonl

# Check dataset size
python scripts/dataset_stats.py

# Minimum recommended: 5,000 training samples
# Ideal: 10,000–20,000
```

---

## Configuration

Edit `src/training/config.py` before training:

```python
# ── Model ─────────────────────────────────────────────
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
YOUR_NAME  = "You"          # Name as it appears in WhatsApp exports

# ── LoRA ──────────────────────────────────────────────
LORA_R          = 16        # Rank. Higher = more expressive, more VRAM
LORA_ALPHA      = 32        # Scaling. Keep at 2x LORA_R
LORA_DROPOUT    = 0.05
TARGET_MODULES  = ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"]

# ── Training ──────────────────────────────────────────
NUM_EPOCHS              = 3
BATCH_SIZE              = 2     # Reduce to 1 if OOM
GRADIENT_ACCUMULATION   = 8     # Effective batch = 16
LEARNING_RATE           = 2e-4
WARMUP_RATIO            = 0.05
LR_SCHEDULER            = "cosine"
MAX_SEQ_LENGTH          = 1024  # Reduce to 512 to save VRAM

# ── Quantization ──────────────────────────────────────
LOAD_IN_4BIT            = True
BNB_4BIT_QUANT_TYPE     = "nf4"
BNB_4BIT_DOUBLE_QUANT   = True   # Saves ~0.5GB extra
COMPUTE_DTYPE           = "float16"
```

---

## Run Training

```bash
conda activate whatsapp-model

# Start training (logs to runs/)
python src/training/train.py

# With specific config overrides
python src/training/train.py --epochs 5 --lr 1e-4

# Monitor GPU usage in another terminal
watch -n 1 nvidia-smi
```

---

## Training Output

```
models/
└── checkpoints/
    ├── checkpoint-500/
    ├── checkpoint-1000/
    └── checkpoint-final/
        ├── adapter_config.json
        ├── adapter_model.safetensors
        └── tokenizer/
```

---

## After Training: Merge & Export

```bash
# Merge LoRA weights into base model
python src/training/merge.py

# Output: models/final/
# This is the model you'll use for inference
```

---

## VRAM Troubleshooting

**OOM during training:**

```python
# In config.py, try these in order:
BATCH_SIZE = 1              # Step 1
MAX_SEQ_LENGTH = 512        # Step 2
LORA_R = 8                  # Step 3 (reduces quality slightly)
GRADIENT_CHECKPOINTING = True  # Step 4 (slower but saves memory)
```

**Check VRAM usage:**
```bash
# During training, in another terminal:
nvidia-smi --query-gpu=memory.used,memory.free --format=csv -l 1
```

---

## Interpreting Training Loss

| Loss value | Meaning |
|---|---|
| > 2.5 | Model hasn't converged yet |
| 1.5 – 2.5 | Learning general patterns |
| 1.0 – 1.5 | Good — style being captured |
| < 1.0 | Great, or possibly overfitting |
| < 0.5 | Likely overfitting — reduce epochs |

Watch for the **validation loss** diverging from training loss — that's overfitting.

---

## Resuming from Checkpoint

```bash
python src/training/train.py --resume models/checkpoints/checkpoint-1000
```

---

## Re-training with New Data

```bash
# Add new .txt exports to data/raw/
# Re-run pipeline
python scripts/run_pipeline.py

# Fine-tune the already-merged model (faster than starting fresh)
python src/training/train.py --base-model models/final
```
