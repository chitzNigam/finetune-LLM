"""
src/training/config.py

Central config for training. Edit these values before running train.py.
"""

import json
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).parent.parent.parent
DATA_DIR        = ROOT_DIR / "data"
MODELS_DIR      = ROOT_DIR / "models"
TRAIN_FILE      = DATA_DIR / "exports" / "train.jsonl"
VAL_FILE        = DATA_DIR / "exports" / "val.jsonl"
CHECKPOINT_DIR  = MODELS_DIR / "checkpoints"
FINAL_DIR       = MODELS_DIR / "final"
CONTACTS_FILE   = DATA_DIR / "contacts.json"


def _load_your_name(default: str = "You") -> str:
    """
    Allow contacts.json to override the sender label used in exports.
    Falls back to the default when the file is missing or invalid.
    """
    if not CONTACTS_FILE.exists():
        return default

    try:
        with open(CONTACTS_FILE, encoding="utf-8") as f:
            config = json.load(f)
    except Exception:
        return default

    return config.get("your_name_in_exports", default)


# ── Identity ───────────────────────────────────────────────────────────────────
# This must match the name that appears in your WhatsApp exports.
# You can also set "your_name_in_exports" in data/contacts.json.
YOUR_NAME = _load_your_name()

# ── Base Model ─────────────────────────────────────────────────────────────────
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
# Alternatives (uncomment to use):
# BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"   # Good English, decent Hindi
# BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"    # Smaller, faster, less expressive
# BASE_MODEL = "Qwen/Qwen2.5-14B-Instruct"          # Better but tight on 16GB

# ── LoRA Config ────────────────────────────────────────────────────────────────
LORA_R          = 16      # Rank. Higher = more expressive. Try 8 if OOM.
LORA_ALPHA      = 32      # Scaling. Keep at 2x LORA_R.
LORA_DROPOUT    = 0.05
TARGET_MODULES  = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# ── Quantization ───────────────────────────────────────────────────────────────
LOAD_IN_4BIT         = True
BNB_4BIT_QUANT_TYPE  = "nf4"        # nf4 > fp4 for LLMs
BNB_4BIT_DOUBLE_QUANT = True        # Saves ~0.5GB extra VRAM
COMPUTE_DTYPE        = "float16"    # bfloat16 if your GPU supports it

# ── Training Hyperparameters ───────────────────────────────────────────────────
NUM_EPOCHS              = 3
BATCH_SIZE              = 2         # Reduce to 1 if OOM
GRADIENT_ACCUMULATION   = 8         # Effective batch = BATCH_SIZE * GRADIENT_ACCUMULATION = 16
LEARNING_RATE           = 2e-4
WARMUP_RATIO            = 0.05
LR_SCHEDULER            = "cosine"
WEIGHT_DECAY            = 0.01
MAX_GRAD_NORM           = 1.0
MAX_SEQ_LENGTH          = 1024      # Reduce to 512 to save VRAM
GRADIENT_CHECKPOINTING  = False     # Set True if VRAM is tight (slower training)
FP16                    = True
BF16                    = False     # Use instead of FP16 if you have Ampere+ GPU

# ── Logging & Saving ───────────────────────────────────────────────────────────
LOGGING_STEPS   = 25
SAVE_STRATEGY   = "epoch"           # Save checkpoint after each epoch
EVAL_STRATEGY   = "epoch"
LOAD_BEST_MODEL = True

# ── Generation (Inference) ─────────────────────────────────────────────────────
GEN_MAX_NEW_TOKENS  = 150
GEN_TEMPERATURE     = 0.85
GEN_TOP_P           = 0.92
GEN_TOP_K           = 50
GEN_REPEAT_PENALTY  = 1.1

# ── Pipeline ───────────────────────────────────────────────────────────────────
CONTEXT_WINDOW_SIZE     = 5         # Messages of context before your reply
SESSION_GAP_SECONDS     = 3600      # 1hr gap = new session
MIN_REPLY_LENGTH        = 2         # Drop replies shorter than this (chars)
MAX_REPLY_LENGTH        = 500       # Drop replies longer than this (chars)
REMOVE_HINDI            = True      # Strip Devanagari script
REMOVE_ENGLISH          = False     # Strip pure English words (conservative)
FILTER_NSFW_ENGLISH     = False     # Replace English NSFW terms with [NSFW]
FILTER_NSFW_HINGLISH    = False     # Replace Hinglish NSFW terms with [NSFW]
FILTER_NSFW_HINDI       = False     # Replace Hindi NSFW terms with [NSFW]
CUTOFF_MONTHS           = 24        # Only use chats from last N months (0 = all)
