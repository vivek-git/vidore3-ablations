"""VRAM profiles for consumer and laptop GPUs."""

from __future__ import annotations

from typing import Any, Dict, Literal

VramProfile = Literal["default", "low_12gb", "ultra_8gb"]

# ~12 GB cards (RTX 3060, 4070 desktop, etc.)
LOW_12GB_PIPELINE_ARGS: Dict[str, Dict[str, Any]] = {
    "dense_text": {"batch_size": 32, "device": "cpu"},
    "clip_visual": {"batch_size": 8},
    "clip_text_on_ocr": {"batch_size": 32},
    "hybrid_rrf": {"batch_size": 8, "text_device": "cpu"},
    "hybrid_weighted": {"batch_size": 8, "text_device": "cpu"},
    "colpali_late_interaction": {
        "batch_size": 1,
        "score_batch_size": 16,
        "score_device": "cpu",
    },
}

# ~8 GB laptop GPUs (RTX 4070 Laptop, 4060, etc.)
ULTRA_8GB_PIPELINE_ARGS: Dict[str, Dict[str, Any]] = {
    "dense_text": {"batch_size": 16, "device": "cpu"},
    "clip_visual": {"batch_size": 4, "image_scale": 0.75},
    "clip_text_on_ocr": {"batch_size": 16, "device": "cpu"},
    "hybrid_rrf": {"batch_size": 4, "text_device": "cpu"},
    "hybrid_weighted": {"batch_size": 4, "text_device": "cpu"},
    "colpali_late_interaction": {
        "batch_size": 1,
        "score_batch_size": 8,
        "score_device": "cpu",
        "max_gpu_memory": "7GiB",
    },
}

PROFILE_ARGS: Dict[VramProfile, Dict[str, Dict[str, Any]]] = {
    "low_12gb": LOW_12GB_PIPELINE_ARGS,
    "ultra_8gb": ULTRA_8GB_PIPELINE_ARGS,
}


def apply_vram_profile_args(
    pipeline_name: str,
    args: Dict[str, Any],
    profile: VramProfile,
) -> Dict[str, Any]:
    if profile == "default":
        return dict(args)
    merged = dict(args)
    merged.update(PROFILE_ARGS.get(profile, {}).get(pipeline_name, {}))
    return merged


def apply_low_vram_args(pipeline_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Backward-compatible alias for the 12 GB profile."""
    return apply_vram_profile_args(pipeline_name, args, "low_12gb")
