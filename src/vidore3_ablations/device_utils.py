"""Device and memory helpers for GPU-constrained environments."""

from __future__ import annotations

import gc
import os
from typing import Literal

import torch

from vidore3_ablations.low_vram import VramProfile

DevicePreference = Literal["auto", "cpu", "cuda", "mps"]


def get_torch_device(preference: DevicePreference = "auto") -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_inference_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def get_cuda_vram_gb(device_index: int = 0) -> float | None:
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(device_index)
    return props.total_memory / (1024**3)


def detect_vram_profile() -> VramProfile:
    vram = get_cuda_vram_gb()
    if vram is None:
        return "default"
    if vram <= 8.5:
        return "ultra_8gb"
    if vram <= 16.0:
        return "low_12gb"
    return "default"


def is_low_vram_gpu(threshold_gb: float = 16.0) -> bool:
    vram = get_cuda_vram_gb()
    return vram is not None and vram <= threshold_gb


def empty_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def configure_pytorch_memory() -> None:
    """Apply allocator settings that reduce fragmentation on consumer GPUs."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.92)


def resolve_score_device(
    encode_device: torch.device,
    preference: DevicePreference,
    corpus_size: int,
    vram_profile: VramProfile = "default",
) -> torch.device:
    """
    MaxSim scoring materializes large query x passage tensors on GPU.
    Keep scoring on CPU for laptop GPUs and large corpora.
    """
    if preference != "auto":
        return get_torch_device(preference)
    if encode_device.type != "cuda":
        return torch.device("cpu")
    if vram_profile == "ultra_8gb":
        return torch.device("cpu")
    if corpus_size > 256 or is_low_vram_gpu(threshold_gb=16.0):
        return torch.device("cpu")
    return encode_device


def release_model(model) -> None:
    del model
    empty_cuda_cache()
