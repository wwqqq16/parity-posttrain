"""Check the local PyTorch runtime and available accelerator."""

from __future__ import annotations

import platform

import torch
import transformers


def select_device() -> torch.device:
    """Select the best available local device."""

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def main() -> None:
    """Print runtime information and run a small tensor operation."""

    device = select_device()

    print("System:", platform.system())
    print("Machine:", platform.machine())
    print("PyTorch:", torch.__version__)
    print("Transformers:", transformers.__version__)
    print("MPS built:", torch.backends.mps.is_built())
    print("MPS available:", torch.backends.mps.is_available())
    print("CUDA available:", torch.cuda.is_available())
    print("Selected device:", device)

    left = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        device=device,
    )
    right = torch.tensor(
        [[2.0, 0.0], [1.0, 2.0]],
        device=device,
    )

    result = left @ right

    if device.type == "mps":
        torch.mps.synchronize()

    print("Matrix result:")
    print(result.cpu())
    print("Runtime check passed.")


if __name__ == "__main__":
    main()
