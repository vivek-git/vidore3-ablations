"""Load and validate ablation YAML configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml

from vidore3_ablations.pipelines.registry import PIPELINE_REGISTRY

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "ablations.yaml"


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_config(config)
    return config


def validate_config(config: Dict[str, Any]) -> None:
    if "ablations" not in config:
        raise ValueError("Config must define an 'ablations' section")

    for name, spec in config["ablations"].items():
        pipeline = spec.get("pipeline")
        if not pipeline:
            raise ValueError(f"Ablation '{name}' is missing a pipeline name")
        if pipeline not in PIPELINE_REGISTRY:
            known = ", ".join(sorted(PIPELINE_REGISTRY))
            raise ValueError(f"Unknown pipeline '{pipeline}' in ablation '{name}'. Known: {known}")


def list_ablation_names(config: Dict[str, Any]) -> List[str]:
    return list(config["ablations"].keys())
