"""
Convert BiSeNetV2 .pth weights into a TorchScript .pt segmentation model.
"""

import argparse
from pathlib import Path
import sys

import torch
import torch.nn as nn

from model.semantic_segmentation.bisenetv2.bisenetv2 import BiSeNetV2


class WrappedBiSeNetV2(nn.Module):
    def __init__(self, model: BiSeNetV2):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        mask = torch.argmax(logits, dim=1, keepdim=True).float()
        return mask


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        if "model" in checkpoint:
            return checkpoint["model"]

    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--weight-path", type=str, required=True)
    parser.add_argument("--out-path", type=str, required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--model-width", type=int, required=True)
    parser.add_argument("--model-height", type=int, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--strict", action="store_true")

    args = parser.parse_args()

    weight_path = Path(args.weight_path)
    out_path = Path(args.out_path)

    if not weight_path.exists():
        raise FileNotFoundError(f"Weight file not found: {weight_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)

    print(f"Using device: {device}")
    print(f"Loading weights: {weight_path}")

    model = BiSeNetV2(
        n_classes=args.num_classes,
        aux_mode="eval",
    )

    checkpoint = torch.load(
        str(weight_path),
        map_location=device,
    )

    state_dict = extract_state_dict(checkpoint)

    load_result = model.load_state_dict(
        state_dict,
        strict=args.strict,
    )

    if not args.strict:
        print(f"Missing keys: {len(load_result.missing_keys)}")
        print(f"Unexpected keys: {len(load_result.unexpected_keys)}")

        if load_result.missing_keys:
            print("First few missing keys:")
            for key in load_result.missing_keys[:10]:
                print(f"  {key}")

        if load_result.unexpected_keys:
            print("First few unexpected keys:")
            for key in load_result.unexpected_keys[:10]:
                print(f"  {key}")

    wrapped_model = WrappedBiSeNetV2(model)

    wrapped_model.to(device)
    wrapped_model.eval()

    example_input = torch.randn(
        1,
        3,
        args.model_height,
        args.model_width,
        device=device,
        dtype=torch.float32,
    )

    print(f"Tracing model with input shape: {tuple(example_input.shape)}")

    with torch.no_grad():
        traced_model = torch.jit.trace(
            wrapped_model,
            example_input,
        )

    traced_model.save(str(out_path))

    print(f"Saved TorchScript model to: {out_path}")


if __name__ == "__main__":
    main()