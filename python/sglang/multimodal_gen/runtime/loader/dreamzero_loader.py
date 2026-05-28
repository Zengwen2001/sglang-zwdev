# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _torch_dtype_from_precision(precision: str | torch.dtype | None) -> torch.dtype:
    if isinstance(precision, torch.dtype):
        return precision
    value = str(precision or "bf16").lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported DreamZero precision: {precision!r}")


def _metadata_path(model_cfg: Any) -> Path:
    explicit = model_cfg.get("metadata_json_path", None)
    if explicit:
        return Path(str(explicit))
    model_path = model_cfg.get("model_path", None)
    if not model_path:
        raise ValueError("DreamZero requires model_path or metadata_json_path.")
    return Path(str(model_path)) / "experiment_cfg" / "metadata.json"


def _validate_metadata_json(model_cfg: Any) -> None:
    path = _metadata_path(model_cfg)
    if not path.is_file():
        raise FileNotFoundError(f"DreamZero metadata json not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"DreamZero metadata json is empty: {path}")
    with open(path, encoding="utf-8") as f:
        json.load(f)


def _build_model_cfg(server_args: Any) -> Any:
    from omegaconf import OmegaConf

    pipeline_config = server_args.pipeline_config
    model_path = getattr(pipeline_config, "model_path", None) or server_args.model_path
    action_horizon = int(getattr(pipeline_config, "action_horizon", 24))
    num_action_per_block = int(
        getattr(pipeline_config, "num_action_per_block", action_horizon)
    )

    cfg = OmegaConf.create(
        {
            "model_type": "dreamzero",
            "precision": getattr(pipeline_config, "precision", "bf16"),
            "model_path": model_path,
            "tokenizer_path": getattr(
                pipeline_config, "tokenizer_path", "/mnt/public/zengwen/umt5-xxl"
            ),
            "metadata_json_path": getattr(pipeline_config, "metadata_json_path", None),
            "embodiment_tag": getattr(pipeline_config, "embodiment_tag", "oxe_droid"),
            "action_horizon": action_horizon,
            "num_action_per_block": num_action_per_block,
            "state_horizon": int(getattr(pipeline_config, "state_horizon", 1)),
            "relative_action": bool(getattr(pipeline_config, "relative_action", False)),
            "relative_action_per_horizon": bool(
                getattr(pipeline_config, "relative_action_per_horizon", False)
            ),
            "relative_action_keys": getattr(
                pipeline_config, "relative_action_keys", None
            )
            or [],
            "target_video_height": int(
                getattr(pipeline_config, "target_video_height", 180)
            ),
            "target_video_width": int(
                getattr(pipeline_config, "target_video_width", 320)
            ),
        }
    )

    from rlinf.models.embodiment.dreamzero.dreamzero_config import (
        validate_dreamzero_sft_model_cfg,
    )

    cfg = validate_dreamzero_sft_model_cfg(cfg)
    if "action_head_cfg" in cfg and "config" in cfg.action_head_cfg:
        cfg.action_head_cfg.config.action_horizon = action_horizon
        cfg.action_head_cfg.config.diffusion_model_cfg.num_action_per_block = (
            num_action_per_block
        )
    cfg.action_horizon = action_horizon
    return cfg


@contextlib.contextmanager
def _patch_tokenizer_regex_warning():
    from transformers import AutoTokenizer

    original = AutoTokenizer.from_pretrained

    def from_pretrained_with_fix(*args, **kwargs):
        name = str(args[0] if args else kwargs.get("pretrained_model_name_or_path", ""))
        lower_name = name.lower()
        if "mistral" in lower_name:
            kwargs.setdefault("fix_mistral_regex", True)
        else:
            kwargs.pop("fix_mistral_regex", None)
        try:
            return original(*args, **kwargs)
        except TypeError as exc:
            if kwargs.get("fix_mistral_regex") and "pre_tokenizer" in str(exc):
                retry_kwargs = dict(kwargs)
                retry_kwargs["fix_mistral_regex"] = False
                return original(*args, **retry_kwargs)
            raise

    AutoTokenizer.from_pretrained = from_pretrained_with_fix
    try:
        yield
    finally:
        AutoTokenizer.from_pretrained = original


def load_dreamzero_policy(server_args: Any) -> tuple[torch.nn.Module, Any]:
    """Load DreamZeroPolicy inside an SGLang multimodal worker process."""

    try:
        from rlinf.models.embodiment.dreamzero import get_model
    except ImportError as exc:
        raise ImportError(
            "DreamZero loader requires RLinf and Groot/DreamZero on PYTHONPATH. "
            "For bench_c.py this is set by --rlinf-root and --dreamzero-root; "
            "real SGLang workers must set the same paths during worker init."
        ) from exc

    model_cfg = _build_model_cfg(server_args)
    _validate_metadata_json(model_cfg)
    dtype = _torch_dtype_from_precision(model_cfg.get("precision", "bf16"))

    logger.info(
        "Loading DreamZero model from %s with tokenizer %s.",
        model_cfg.get("model_path"),
        model_cfg.get("tokenizer_path"),
    )
    with _patch_tokenizer_regex_warning():
        model = get_model(model_cfg, torch_dtype=dtype)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device=device)
    model.eval()
    logger.info("DreamZero model is ready on %s with dtype=%s.", device, dtype)
    return model, model_cfg
