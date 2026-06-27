# Inference Guide

Running your trained model locally.

---

## Two Inference Modes

| Mode | Tool | Best For |
|---|---|---|
| **CLI Chat** | Python script | Quick testing, development |
| **Ollama API** | Ollama | Clean local API, integrations |

---

## Mode 1: CLI Chat

```bash
conda activate whatsapp-model
python src/inference/chat.py
```

Interactive prompt:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  WhatsApp Style Model — CLI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Recipient name   : Rahul
Relationship     : best_friend
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Rahul]: kal milte hain kya
[Rahul]: bata na yaar
[You]  : █
```

Type the incoming messages, press Enter twice to generate your reply.

---

## Mode 2: Ollama Deployment

### Convert to GGUF and load in Ollama

```bash
# export_gguf.py expects llama.cpp tools and will install/clone them if missing.
# You do not need llama-cpp-python for this path.

# Convert merged model to GGUF (Q4_K_M quantization)
python scripts/export_gguf.py

# Create Ollama modelfile
cat > models/final/Modelfile << 'EOF'
FROM ./model-q4_k_m.gguf

SYSTEM """
You are replicating a person's WhatsApp texting style.
You will be given conversation context and metadata.
Reply naturally, matching the style implied by the relationship tag.
Keep replies concise — this is texting, not an essay.
"""

PARAMETER temperature 0.85
PARAMETER top_p 0.92
PARAMETER top_k 50
PARAMETER repeat_penalty 1.1
PARAMETER num_predict 150
EOF

# Register with Ollama
ollama create whatsapp-me -f models/final/Modelfile

# Test
ollama run whatsapp-me
```

### Query via API

```python
import requests

def generate_reply(context, relationship, hour, timedelta_sec):
    prompt = f"""Relationship: {relationship}
Hour: {hour}, TimeDelta: {timedelta_sec}s

{context}

Reply:"""

    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "whatsapp-me",
        "prompt": prompt,
        "stream": False
    })
    return response.json()["response"]

# Example
reply = generate_reply(
    context="Rahul: kal milte hain kya\nRahul: bata na yaar",
    relationship="best_friend",
    hour=22,
    timedelta_sec=3600
)
print(reply)
```

---

## Generation Parameters Explained

```python
temperature = 0.85   # Controls randomness. 
                     # 0.7 = more consistent, 1.0 = more varied
                     # Your texting has natural variation — keep ~0.85

top_p = 0.92         # Nucleus sampling. Filters unlikely tokens.
                     # 0.9–0.95 is the sweet spot for chat style

top_k = 50           # Limits vocab pool per token. 
                     # 40–60 works well for informal text

repeat_penalty = 1.1 # Prevents looping/repetition
                     # Increase to 1.2 if you see repetitive output

max_new_tokens = 150 # Your replies are short. 150 tokens ≈ 2–3 sentences
                     # Reduce to 80 for very terse reply style
```

---

## Batch Inference (for evaluation)

```bash
# Generate replies for a test set and compare to real replies
python src/inference/evaluate.py \
    --model models/final \
    --n 100 \
    --output results/eval_output.json

# View results
python scripts/show_eval.py results/eval_output.json
```

---

## VRAM Usage at Inference

With merged 4-bit model:
```
Model weights:    ~6.0 GB
KV cache:         ~1.0 GB
Activations:      ~0.5 GB
─────────────────────────
Total:            ~7.5 GB   ✅ Comfortable on 16GB
```

You can run inference and have browser/other apps open simultaneously.

---

## Inference Speed (Expected)

On RTX 4070 Ti Super with 4-bit quantized Qwen2.5-7B:

| Tokens | Time |
|---|---|
| 50 tokens | ~1.2s |
| 100 tokens | ~2.0s |
| 150 tokens | ~2.8s |

Fast enough for real-time use.
