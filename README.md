# 💬 WhatsApp Style Model

> Fine-tune a local LLM to imitate **your** texting style — per relationship, per context, fully offline.

---

## What This Does

Trains a quantized LLM (Qwen2.5-7B) on your WhatsApp exports to generate replies that sound like **you** — adapting tone for friends, family, girlfriend, colleagues, and strangers.

**Everything runs locally. Your chats never leave your machine.**

Cleaning is configurable: you can strip Hindi, strip conservative English filler words, and optionally replace NSFW words in English, Hinglish, and Hindi before training.

---

## Quick Start

```bash
# 1. Clone and setup
git clone <your-repo>
cd whatsapp-style-model
./scripts/setup_wsl.sh

# 2. Export your WhatsApp chats (see docs/EXPORT_GUIDE.md)
# Drop .txt files into data/raw/
# Android note: large chats may be truncated by WhatsApp export.
# For full history on Android, use the local backup + key workflow in docs/EXPORT_GUIDE.md.

# 3. Tag your contacts
cp data/contacts.example.json data/contacts.json
# Edit contacts.json with your contact → relationship mappings

# 4. Run the full pipeline
python scripts/run_pipeline.py

# 5. Train
python src/training/train.py

# 6. Inference
python src/inference/chat.py
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | RTX 3080 (10GB) | **RTX 4070 Ti Super (16GB)** |
| RAM | 16GB | 32GB |
| Disk | 40GB free | 80GB free |
| OS | WSL2 (Ubuntu 22.04) | WSL2 (Ubuntu 22.04) |

---

## Project Structure

```
whatsapp-style-model/
├── data/
│   ├── raw/                  # Drop your WhatsApp .txt exports here
│   ├── processed/            # Cleaned, parsed dataframes (.parquet)
│   └── exports/              # Final .jsonl training datasets
├── src/
│   ├── parser/               # WhatsApp .txt → DataFrame
│   ├── features/             # Feature engineering + context building
│   ├── training/             # QLoRA fine-tuning
│   └── inference/            # Local inference + CLI chat
├── scripts/
│   ├── setup_wsl.sh          # One-shot WSL2 environment setup
│   └── run_pipeline.py       # End-to-end data → dataset pipeline
├── models/
│   ├── checkpoints/          # Training checkpoints (gitignored)
│   └── final/                # Merged model for deployment (gitignored)
├── notebooks/                # EDA and evaluation notebooks
├── docs/                     # Detailed documentation
└── tests/                    # Unit tests
```

---

## Documentation

| Doc | Description |
|---|---|
| [docs/EXPORT_GUIDE.md](docs/EXPORT_GUIDE.md) | Android/iOS export guide, plus Android backup fallback for full history |
| [docs/SETUP_WSL.md](docs/SETUP_WSL.md) | WSL2 + CUDA setup from scratch |
| [docs/PIPELINE.md](docs/PIPELINE.md) | Data pipeline explained |
| [docs/TRAINING.md](docs/TRAINING.md) | Training configuration and tips |
| [docs/INFERENCE.md](docs/INFERENCE.md) | Running inference locally |
| [docs/PRIVACY.md](docs/PRIVACY.md) | Privacy and data security |

---

## Pipeline Overview

```
WhatsApp .txt exports
        ↓
   Parse & Deduplicate          (src/parser/)
        ↓
   Clean Text                   (src/features/cleaner.py)
        ↓
   Engineer Features            (src/features/engineer.py)
        ↓
   Build Context Windows        (src/features/context.py)
        ↓
   Format as ChatML .jsonl      (src/features/context.py)
        ↓
   QLoRA Fine-tune              (src/training/train.py)
        ↓
   Merge + Export               (src/training/merge.py)
        ↓
   Local Inference              (src/inference/chat.py)
```

---

## Privacy

- ✅ 100% local — no data leaves your machine
- ✅ Train offline — no internet required after model download
- ✅ Inference offline — via Ollama or llama.cpp
- ⚠️ Raw chat data is sensitive — see [docs/PRIVACY.md](docs/PRIVACY.md)

---

## License

Personal use only. Do not distribute model weights trained on private conversations.
