"""
scripts/verify_setup.py

Checks all dependencies and GPU setup are working correctly.
"""

import sys


def check(label, fn):
    try:
        result = fn()
        print(f"  ✅ {label}" + (f"  —  {result}" if result else ""))
        return True
    except Exception as e:
        print(f"  ❌ {label}  —  {e}")
        return False


def main():
    print("\nVerifying setup...\n")
    all_ok = True

    all_ok &= check("Python 3.11+", lambda: sys.version.split()[0])

    all_ok &= check("torch", lambda: __import__('torch').__version__)

    def check_cuda():
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory // (1024**3)
        return f"{name} ({mem} GB)"
    all_ok &= check("CUDA + GPU", check_cuda)

    for pkg in ['transformers', 'peft', 'bitsandbytes', 'trl', 'datasets',
                'sklearn', 'pandas', 'pyarrow']:
        all_ok &= check(pkg, lambda p=pkg: __import__(p).__version__)

    def check_ollama():
        import subprocess
        result = subprocess.run(['ollama', 'list'], capture_output=True, timeout=3)
        return "running" if result.returncode == 0 else "not running"
    check("ollama", check_ollama)  # Non-fatal

    print()
    if all_ok:
        print("  All checks passed! You're ready to train.\n")
    else:
        print("  Some checks failed. See docs/SETUP_WSL.md for help.\n")


if __name__ == "__main__":
    main()
