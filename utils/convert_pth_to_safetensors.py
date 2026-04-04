"""
One-off conversion: model.pth -> model.safetensors so HuggingFace can load
without torch.load (avoids transformers' torch>=2.6 requirement for .pth).
Run from repo root: python utils/convert_pth_to_safetensors.py <model_dir>
"""
import argparse
import sys
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert model.pth to model.safetensors")
    parser.add_argument("model_dir", type=Path, help="Directory containing config.json and model.pth")
    args = parser.parse_args()
    model_dir = args.model_dir.resolve()
    pth = model_dir / "model.pth"
    out = model_dir / "model.safetensors"
    if not pth.exists():
        print(f"Not found: {pth}", file=sys.stderr)
        sys.exit(1)
    if out.exists():
        print(f"Already exists: {out}", file=sys.stderr)
        sys.exit(0)
    print(f"Loading {pth} ...")
    try:
        state_dict = torch.load(str(pth), map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(str(pth), map_location="cpu")
    # Safetensors only supports tensors; drop any other values
    tensors = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
    if len(tensors) != len(state_dict):
        print(f"Skipped {len(state_dict) - len(tensors)} non-tensor entries.")
    print(f"Saving {len(tensors)} tensors to {out} ...")
    try:
        from safetensors.torch import save_file
    except ImportError:
        print("Install safetensors: pip install safetensors", file=sys.stderr)
        sys.exit(1)
    save_file(tensors, str(out))
    print("Done.")


if __name__ == "__main__":
    main()
