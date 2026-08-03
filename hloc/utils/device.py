import os
import sys

import torch

from .. import logger


def get_device() -> str:
    """Select the compute device.

    Priority: HLOC_DEVICE env var > CUDA > Apple Silicon (MPS) > CPU.
    """
    override = os.environ.get("HLOC_DEVICE")
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def to_device(tensor: torch.Tensor, device: str) -> torch.Tensor:
    """Move a tensor to the device, casting dtypes that MPS does not support."""
    if device == "mps" and tensor.dtype == torch.float64:
        tensor = tensor.float()
    # non_blocking is only meaningful for pinned CPU -> CUDA transfers.
    return tensor.to(device, non_blocking=(device == "cuda"))


def dataloader_kwargs(num_workers: int, device: str) -> dict:
    """DataLoader kwargs tuned for the platform.

    macOS starts workers with `spawn`, which re-imports torch in every child
    and costs ~5 s per worker. That dwarfs the loading work itself here
    (<10 ms/item), so we default to in-process loading. Override with
    HLOC_NUM_WORKERS.
    """
    override = os.environ.get("HLOC_NUM_WORKERS")
    if override is not None:
        num_workers = int(override)
    elif sys.platform == "darwin":
        num_workers = 0
    return {
        "num_workers": num_workers,
        # pin_memory only helps pinned CPU -> CUDA copies.
        "pin_memory": device == "cuda",
    }


_warned = False


def warn_mps_fallback():
    """MPS lacks a few ops; make sure the CPU fallback is enabled."""
    global _warned
    if not _warned and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
        logger.warning(
            "Running on MPS without PYTORCH_ENABLE_MPS_FALLBACK=1: "
            "unsupported ops will raise instead of falling back to CPU."
        )
        _warned = True
