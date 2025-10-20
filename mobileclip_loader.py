"""Shared helpers for loading MobileCLIP2-S2 models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import open_clip
import torch
from PIL import Image

from mobileclip.modules.common.mobileone import reparameterize_model
from open_clip import factory as open_clip_factory

logger = logging.getLogger(__name__)

MODEL_NAME = "MobileCLIP2-S2"
HF_MODEL_DIR = Path("/mnt/data/huggingface/hub/models--apple--MobileCLIP2-S2")


def _ensure_model_configs_registered() -> Path:
    config_dir = Path(__file__).resolve().parent / "mobileclip2" / "model_configs"
    if not config_dir.exists():
        raise FileNotFoundError(f"Model config directory missing: {config_dir}")
    open_clip_factory.add_model_config(config_dir)
    return config_dir


def _resolve_device(device: Optional[str | torch.device] = None) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def _get_pretrained_path() -> Path:
    snapshot_id_path = HF_MODEL_DIR / "refs" / "main"
    if not snapshot_id_path.exists():
        raise FileNotFoundError(f"Hugging Face snapshot ref missing: {snapshot_id_path}")
    snapshot_id = snapshot_id_path.read_text().strip()
    pretrained_path = HF_MODEL_DIR / "snapshots" / snapshot_id / "mobileclip2_s2.pt"
    if not pretrained_path.exists():
        raise FileNotFoundError(f"MobileCLIP2 weights not found at {pretrained_path}")
    return pretrained_path


def load_mobileclip2_s2(
    *,
    device: Optional[str | torch.device] = None,
) -> Tuple[torch.nn.Module, Callable[[Image.Image], torch.Tensor], Any, torch.device]:
    resolved_device = _resolve_device(device)
    logger.info("Loading %s on %s", MODEL_NAME, resolved_device)

    _ensure_model_configs_registered()
    pretrained_path = _get_pretrained_path()

    model_kwargs = {}
    if not (
        MODEL_NAME.endswith("S3")
        or MODEL_NAME.endswith("S4")
        or MODEL_NAME.endswith("L-14")
    ):
        model_kwargs = {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=str(pretrained_path),
        **model_kwargs,
    )
    model = reparameterize_model(model)
    model.eval()
    model.to(resolved_device)

    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    return model, preprocess, tokenizer, resolved_device

