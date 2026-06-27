# Privacy & Data Security

This project handles extremely sensitive data — private conversations between you and real people who have not consented to their messages being used for ML training. Take this seriously.

---

## Core Principles

1. **Train locally** — model never sees an external API
2. **Infer locally** — responses generated on your GPU only
3. **Delete raw data after training** — weights don't store verbatim messages
4. **Never commit chat data** — `.gitignore` covers this, but double-check

---

## What Data is Sensitive

| Data | Sensitivity | Action |
|---|---|---|
| Raw `.txt` exports | 🔴 Extremely high | Encrypt or delete after processing |
| Processed `.parquet` files | 🔴 High | Encrypt, never commit |
| `.jsonl` training dataset | 🟠 High | Encrypt, never commit |
| Model weights (`.safetensors`) | 🟡 Medium | Contains style, not verbatim text |
| `contacts.json` | 🟡 Medium | Contains phone numbers |

---

## Encrypting Your Data Directory

```bash
# Install VeraCrypt (on Windows) or use built-in Linux encryption
# Simpler option: encrypt the data folder with GPG

# Compress and encrypt
tar -czf data.tar.gz data/
gpg --symmetric --cipher-algo AES256 data.tar.gz
rm -rf data/ data.tar.gz  # delete unencrypted version

# Decrypt when you need to work
gpg --decrypt data.tar.gz.gpg | tar -xzf -
```

---

## .gitignore Rules (Already Configured)

```
data/raw/
data/processed/
data/exports/
models/
*.parquet
*.jsonl
contacts.json
```

**Never override these.** If you fork this repo, verify these rules are intact before pushing.

---

## Stripping PII from Message Text

Phone numbers, emails, and addresses in message text are stripped during cleaning:

```python
# src/features/cleaner.py handles:
# - Phone numbers → [PHONE]
# - Email addresses → [EMAIL]
# - UPI IDs → [UPI]
# - Aadhaar-like numbers → [ID]
```

This happens automatically in the pipeline.

---

## After Training

Once training is complete and you're satisfied with the model:

```bash
# Delete raw and processed data, keep only model weights
python scripts/cleanup_data.py

# This removes:
# - data/raw/*.txt
# - data/processed/*.parquet
# - data/exports/*.jsonl
# Keeps: models/final/ (the trained weights)
```

---

## Regarding Other People's Messages

You are training on messages that **other people sent you**. Their messages form the context window — the model learns what kind of inputs lead to your responses. Ethically:

- Don't share or publish the trained model
- Don't use it to impersonate yourself in ways that could deceive or harm others
- Don't deploy it in any public-facing product

This model is a **personal productivity tool**, not a product.
