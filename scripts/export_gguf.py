"""
scripts/export_gguf.py

Converts the merged HuggingFace model to GGUF format for Ollama / llama.cpp.
Requires llama.cpp to be installed (handled in setup_wsl.sh).

Usage:
    python scripts/export_gguf.py
    python scripts/export_gguf.py --quant q4_k_m   (default, best quality/size)
    python scripts/export_gguf.py --quant q5_k_m   (slightly better, larger)
    python scripts/export_gguf.py --quant q8_0     (best quality, 8-bit)
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.training.config import MODELS_DIR, YOUR_NAME


QUANT_OPTIONS = {
    'q4_k_m': 'Recommended — best balance of quality and size (~4.5 GB)',
    'q5_k_m': 'Higher quality, slightly larger (~5.5 GB)',
    'q8_0':   'Best quality, largest (~8 GB)',
    'q4_0':   'Smallest, lowest quality (~3.5 GB)',
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--quant',  type=str, default='q4_k_m',
                   choices=list(QUANT_OPTIONS.keys()))
    p.add_argument('--model',  type=str, default=str(MODELS_DIR / "final"))
    p.add_argument('--output', type=str, default=str(MODELS_DIR / "final"))
    return p.parse_args()


def check_llama_cpp():
    """Check if llama.cpp convert script is available."""
    candidates = [
        Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
        Path("/usr/local/bin/convert_hf_to_gguf.py"),
    ]
    for path in candidates:
        if path.exists():
            return path

    print("[WARN] llama.cpp not found.")
    print("[WARN] Cloning llama.cpp from GitHub and installing its requirements.")
    print("[WARN] Review this path before running on a machine with private chat data.")
    subprocess.run([
        "git", "clone", "--depth=1",
        "https://github.com/ggerganov/llama.cpp.git",
        str(Path.home() / "llama.cpp")
    ], check=True)
    subprocess.run([
        "pip", "install", "-r",
        str(Path.home() / "llama.cpp" / "requirements.txt")
    ], check=True)
    return Path.home() / "llama.cpp" / "convert_hf_to_gguf.py"


def main():
    args = parse_args()
    model_path  = Path(args.model)
    output_path = Path(args.output)

    if not model_path.exists():
        print(f"[ERROR] Model not found at {model_path}")
        print("Run: python src/training/merge.py first")
        sys.exit(1)

    print(f"\n[INFO] Quantization: {args.quant}  —  {QUANT_OPTIONS[args.quant]}")

    convert_script = check_llama_cpp()
    gguf_fp16      = output_path / "model-f16.gguf"
    gguf_quant     = output_path / f"model-{args.quant}.gguf"

    # Step 1: Convert to FP16 GGUF
    print(f"\n[1/2] Converting to FP16 GGUF...")
    subprocess.run([
        sys.executable, str(convert_script),
        str(model_path),
        "--outfile", str(gguf_fp16),
        "--outtype", "f16",
    ], check=True)

    # Step 2: Quantize
    print(f"\n[2/2] Quantizing to {args.quant}...")
    llama_quantize = Path.home() / "llama.cpp" / "llama-quantize"
    if not llama_quantize.exists():
        # Try to build
        subprocess.run(["make", "-C", str(Path.home() / "llama.cpp"),
                        "llama-quantize", "-j4"], check=True)

    subprocess.run([
        str(llama_quantize),
        str(gguf_fp16),
        str(gguf_quant),
        args.quant.upper(),
    ], check=True)

    # Clean up FP16 intermediate
    gguf_fp16.unlink(missing_ok=True)

    # Create Ollama Modelfile
    modelfile_path = output_path / "Modelfile"
    modelfile_path.write_text(f"""FROM ./{gguf_quant.name}

SYSTEM \"\"\"
You are replicating {YOUR_NAME}'s WhatsApp texting style exactly.
You will receive recent conversation messages and contextual metadata.
Generate a reply that matches {YOUR_NAME}'s natural texting style for the given relationship.
Keep it authentic — match their typical length, tone, vocabulary, and emoji usage.
This is texting, not formal writing. Be natural.
\"\"\"

PARAMETER temperature 0.85
PARAMETER top_p 0.92
PARAMETER top_k 50
PARAMETER repeat_penalty 1.1
PARAMETER num_predict 150
""")

    print(f"\n✅ GGUF export complete!")
    print(f"   Model:     {gguf_quant}")
    print(f"   Size:      {gguf_quant.stat().st_size / 1e9:.2f} GB")
    print(f"   Modelfile: {modelfile_path}")
    print(f"\nRegister with Ollama:")
    print(f"   cd {output_path}")
    print(f"   ollama create whatsapp-me -f Modelfile")
    print(f"   ollama run whatsapp-me\n")


if __name__ == "__main__":
    main()
