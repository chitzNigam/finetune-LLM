# WhatsApp Chat Export Guide

How to export your chats from Android and iOS, and where to put them.

---

## Android

For small and medium chats, use WhatsApp's built-in export. For large group chats, WhatsApp may only export the most recent portion of the conversation. If that happens, the practical fallback on Android is to extract your own local backup database and convert it separately.

1. Open WhatsApp → open any chat
2. Tap ⋮ (three dots) → **More** → **Export chat**
3. Choose **Without Media**
4. Share to yourself via Telegram, Gmail, or save to Files
5. Copy the `.txt` file to `data/raw/` in your WSL home:

```bash
# From Windows, copy to WSL:
cp /mnt/c/Users/<YourName>/Downloads/WhatsApp\ Chat\ with\ Rahul.txt ~/whatsapp-style-model/data/raw/
```

Repeat for every contact you want to include.

### Android Full-History Fallback

Use this when a `.txt` export is truncated and does not include the full chat history.

What you need:

- Your own Android device
- Access to the local WhatsApp backup files
- The backup key for that backup

Typical Android backup file locations:

```text
/Android/media/com.whatsapp/WhatsApp/Databases/
```

Common files:

```text
msgstore.db.crypt14
msgstore.db.crypt15
msgstore-increment-*.db.crypt14
msgstore-increment-*.db.crypt15
wa.db.crypt14
wa.db.crypt15
```

Important limits:

- The built-in parser in this repo only reads WhatsApp `.txt` exports today.
- Decrypting backup databases is a separate pre-processing step.
- If you do not have the matching backup key, the encrypted backup is not practically usable.
- If you use end-to-end encrypted backups, save the 64-digit key when WhatsApp shows it.

Recommended workflow for future-proofing:

1. Export chats as `.txt` when possible.
2. If a chat is truncated, copy the local `msgstore*.crypt*` files off your Android device.
3. Save the 64-digit backup key or key file securely at the same time.
4. Decrypt the database to SQLite.
5. Convert the decrypted data to `.txt` or a structured intermediate format before feeding it into this repo.

---

## iOS

1. Open WhatsApp → open any chat
2. Tap the contact name at the top → **Export Chat**
3. Choose **Without Media**
4. AirDrop to Mac, or email to yourself
5. Transfer the `.txt` to your WSL `data/raw/` folder

---

## File Naming Convention

Rename your exported files to make contact tagging easier:

```
data/raw/
├── rahul_bestfriend.txt
├── mom_family.txt
├── priya_girlfriend.txt
├── ankit_closefriend.txt
├── dr_sharma_colleague.txt
└── college_group_group.txt
```

Format: `<name>_<hint>.txt` — the hint is optional but helps during `contacts.json` setup.

---

## How Many Chats to Export

| Data size | Expected quality |
|---|---|
| < 3,000 your messages | Poor — not enough signal |
| 3,000 – 8,000 | Okay — general style learned |
| 8,000 – 20,000 | Good — relationship styles emerge |
| 20,000+ | Excellent — nuanced per-person patterns |

**Priority order:** Export your most-texted contacts first. 80% of your messages are probably with 5–10 people.

---

## WhatsApp Export Format

The parser expects standard WhatsApp export format:

```
DD/MM/YYYY, HH:MM - Name: message text
```

Example:
```
15/03/2024, 22:14 - Rahul: kal milte hain kya
15/03/2024, 22:15 - You: haan bhai, evening chalega
15/03/2024, 22:15 - Rahul: 👍
```

> **Note:** "You" or your saved name is used for your messages. Check what name appears in one of your exports and set `YOUR_NAME` in `config.py`.

---

## Group Chats

Group chats are supported but handled differently:

- Your messages are still extracted as training samples
- The "recipient" is tagged as `group_close` or `group_formal`
- Context window includes all senders (not just one person)

For v1, you can skip group chats entirely by not placing them in `data/raw/`. Recommended unless you have very few individual chats.

If a busy group chat is missing older messages in the `.txt` export, that is usually a WhatsApp export-size limitation rather than a parser bug in this repo.

---

## After Exporting

```bash
# Check how many files you have
ls data/raw/*.txt | wc -l

# Quick preview of parsed count (before full pipeline)
python scripts/preview_data.py
```
