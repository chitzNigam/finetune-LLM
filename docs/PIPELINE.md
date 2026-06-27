# Data Pipeline

Detailed explanation of each pipeline stage.

---

## Run the Full Pipeline

```bash
python scripts/run_pipeline.py
```

Or run stages individually:

```bash
python scripts/run_pipeline.py --stage parse
python scripts/run_pipeline.py --stage clean
python scripts/run_pipeline.py --stage features
python scripts/run_pipeline.py --stage format
```

---

## Stage 1: Parse

**Input:** `data/raw/*.txt`
**Output:** `data/processed/parsed.parquet`

Converts raw WhatsApp exports into a structured DataFrame:

```
datetime | sender | text | source_file
```

Handles:
- Multi-line messages (WhatsApp wraps long messages)
- System messages (`"Messages and calls are end-to-end encrypted"`) → dropped
- Media placeholders (`<Media omitted>`) → flagged, not dropped
- Date format variations (DD/MM/YYYY and MM/DD/YYYY)

---

## Stage 2: Clean Text

**Input:** `data/processed/parsed.parquet`
**Output:** `data/processed/cleaned.parquet`

Text transformations:
- Remove Devanagari script (`[\u0900-\u097F]+`)
- Remove pure English words (via wordlist) — configurable
- Replace NSFW words in English, Hinglish, and Hindi with `[NSFW]` — configurable
- Replace URLs → `[URL]`
- Replace phone numbers → `[PHONE]`
- Replace emails → `[EMAIL]`
- Collapse whitespace
- **Preserve:** emojis, punctuation style, capitalization, typos

> ⚙️ Configure what gets stripped in `src/features/cleaner.py`

---

## Stage 3: Tag & Enrich

**Input:** `data/processed/cleaned.parquet` + `data/contacts.json`
**Output:** `data/processed/enriched.parquet`

Adds:
- `recipient_phone` — from contacts.json mapping
- `relationship` — from contacts.json mapping
- `conversation_id` — hash of chat participants
- `timedelta_prev` — seconds since previous message
- `session_id` — breaks conversation into sessions (gap > 1hr = new session)
- `position` — `opener` / `mid` / `closer`
- `hour`, `day_of_week`, `is_weekend`
- `reply_chain_depth` — turns in current session

---

## Stage 4: Build Context Windows

**Input:** `data/processed/enriched.parquet`
**Output:** `data/processed/contexts.parquet`

For each of your messages, collects the preceding N messages as context:

```
context_window = last 5 messages before your reply
target = your reply text
```

Respects session boundaries, so context does not bleed across long gaps.

---

## Stage 5: Format Dataset

**Input:** `data/processed/contexts.parquet`
**Output:** `data/exports/train.jsonl`, `data/exports/val.jsonl`, `data/exports/test.jsonl`

Formats each sample into ChatML instruction format:

```json
{
  "text": "<|im_start|>system\nYou are replicating...\n<|im_end|>\n<|im_start|>user\n...\n<|im_end|>\n<|im_start|>assistant\n...<|im_end|>"
}
```

Split: **85% train / 10% val / 5% test** — stratified by relationship.

---

## contacts.json Format

```json
{
  "your_name_in_exports": "You",
  "contacts": {
    "+919876543210": {
      "name": "Rahul",
      "relationship": "best_friend"
    },
    "+919123456789": {
      "name": "Mom",
      "relationship": "family"
    },
    "+919000000001": {
      "name": "Priya",
      "relationship": "girlfriend"
    },
    "+919000000002": {
      "name": "Ankit",
      "relationship": "close_friend"
    },
    "+919000000003": {
      "name": "Dr Sharma",
      "relationship": "colleague"
    }
  },
  "group_chats": {
    "College Group": "group_close",
    "Work Team": "group_formal"
  },
  "default_relationship": "acquaintance"
}
```

Valid relationship values:
```
best_friend | close_friend | girlfriend | family | colleague | acquaintance | stranger | group_close | group_formal
```

---

## Dataset Statistics

After running the pipeline:

```bash
python scripts/dataset_stats.py
```

Sample output:
```
Total training samples:    12,847
Validation samples:         1,511
Test samples:                 756

By relationship:
  best_friend     4,210  (32.7%)
  family          2,891  (22.5%)
  close_friend    2,103  (16.4%)
  girlfriend      1,876  (14.6%)
  colleague         892  ( 6.9%)
  acquaintance      875  ( 6.8%)

Avg reply length:      23.4 tokens
Avg context length:    87.2 tokens
Date range:            2022-01-15 → 2024-11-30
```
