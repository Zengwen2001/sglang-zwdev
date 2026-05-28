# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from sglang.multimodal_gen.configs.pipeline_configs.dreamzero import (
    DreamZeroPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.dreamzero import DreamZeroSamplingParams
from sglang.multimodal_gen.runtime.loader.dreamzero_loader import load_dreamzero_policy

logger = logging.getLogger(__name__)


def _as_tensor(value: Any) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    return torch.as_tensor(value)


class DreamZeroDenoiseStage:
    """Stage-direct DreamZero action inference.

    C-base intentionally calls RLinf DreamZeroPolicy.predict_action_batch so
    transforms, unapply, and rollout-compatible action formatting stay aligned
    with the B0/B1 baselines. It does not claim to exercise SGLang's executor,
    scheduler, or disaggregated worker stack.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model

    def _extract_env_obs(self, batch: Any) -> dict[str, Any]:
        obs = batch.extra.get("dreamzero_obs")
        if obs is None and batch.sampling_params is not None:
            obs = getattr(batch.sampling_params, "dreamzero_obs", None)
        if obs is None:
            raise ValueError(
                "DreamZero request requires env obs in Req.extra['dreamzero_obs'] "
                "or sampling_params.dreamzero_obs."
            )

        env_obs = dict(obs)
        for key in ("main_images", "wrist_images", "extra_view_images", "states"):
            if key in env_obs:
                env_obs[key] = _as_tensor(env_obs[key])
        if "task_descriptions" in env_obs and env_obs["task_descriptions"] is not None:
            env_obs["task_descriptions"] = list(env_obs["task_descriptions"])
        return env_obs

    def forward(self, batch: Any, server_args: Any) -> Any:
        env_obs = self._extract_env_obs(batch)
        with torch.inference_mode():
            actions, result = self.model.predict_action_batch(env_obs, mode="eval")

        action_tensor = torch.as_tensor(actions, dtype=torch.float32)
        batch.output = action_tensor
        batch.extra["dreamzero_actions"] = action_tensor
        batch.extra["dreamzero_result"] = result
        return batch


class DreamZeroPipeline:
    pipeline_name = "DreamZeroPipeline"
    pipeline_config_cls = DreamZeroPipelineConfig
    sampling_params_cls = DreamZeroSamplingParams

    def __init__(
        self,
        model_path: str,
        server_args: Any,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
        **_: Any,
    ):
        self.model_path = model_path
        self.server_args = server_args
        self._stages: list[Any] = []
        self._stage_name_mapping: dict[str, Any] = {}
        self.modules = self.load_modules(server_args, loaded_modules)
        self.initialize_pipeline(server_args)
        self.create_pipeline_stages(server_args)

    @property
    def stages(self) -> list[Any]:
        return self._stages

    def add_stage(self, stage: Any, stage_name: str | None = None) -> None:
        self._stages.append(stage)
        if stage_name is not None:
            self._stage_name_mapping[stage_name] = stage

    def load_modules(
        self,
        server_args: Any,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, Any]:
        if loaded_modules is not None and "dreamzero_model" in loaded_modules:
            model = loaded_modules["dreamzero_model"]
            model_cfg = None
        else:
            model, model_cfg = load_dreamzero_policy(server_args)

        modules: dict[str, Any] = {
            "dreamzero_model": model,
            "dreamzero_model_cfg": model_cfg,
        }
        action_head_model = getattr(getattr(model, "action_head", None), "model", None)
        if action_head_model is not None and hasattr(action_head_model, "blocks"):
            modules["transformer_blocks"] = action_head_model.blocks
        return modules

    def initialize_pipeline(self, server_args: Any):
        model = self.modules["dreamzero_model"]
        if server_args.enable_cfg_parallel:
            raise NotImplementedError(
                "DreamZeroPipeline C-base is stage-direct and does not implement "
                "SGLang CFG parallel. Use the C4 Groot/SGLang mesh path."
            )
        logger.info("DreamZeroPipeline initialized in stage-direct C-base mode.")
        return model

    def create_pipeline_stages(self, server_args: Any) -> None:
        self.add_stage(
            DreamZeroDenoiseStage(self.modules["dreamzero_model"]),
            stage_name="dreamzero_denoise",
        )

    def forward(self, batch: Any, server_args: Any) -> Any:
        for stage in self._stages:
            batch = stage.forward(batch, server_args)
        return batch


EntryClass = DreamZeroPipeline
