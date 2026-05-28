# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from enum import Enum, auto


class ModelTaskType(Enum):
    I2V = auto()


@dataclass
class ModelDeploymentConfig:
    auto_dit_layerwise_offload: bool = False


@dataclass
class DreamZeroPipelineConfig:
    """Stage-direct DreamZero action inference config.

    C-base keeps this config independent from SGLang's generic pipeline config
    import chain so the DreamZero adapter can be reviewed without pulling in
    unrelated quantization/deep_gemm modules.
    """

    task_type: ModelTaskType = ModelTaskType.I2V
    model_path: str = ""

    tokenizer_path: str = "/mnt/public/zengwen/umt5-xxl"
    metadata_json_path: str | None = None
    embodiment_tag: str = "oxe_droid"

    precision: str = "bf16"
    action_horizon: int = 24
    num_action_per_block: int = 24
    state_horizon: int = 1

    target_video_height: int = 180
    target_video_width: int = 320

    relative_action: bool = False
    relative_action_per_horizon: bool = False
    relative_action_keys: list[str] | None = None

    enable_cfg_parallel: bool = False

    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        return ModelDeploymentConfig(auto_dit_layerwise_offload=False)
