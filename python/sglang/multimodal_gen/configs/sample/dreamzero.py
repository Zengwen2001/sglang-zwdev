# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class DataType(Enum):
    VIDEO = auto()


@dataclass
class DreamZeroSamplingParams:
    """Stage-direct sampling params for DreamZero action inference."""

    data_type: DataType = DataType.VIDEO
    num_inference_steps: int = 4
    guidance_scale: float = 1.0
    cfg_scale: float = 5.0

    action_horizon: int = 24
    num_action_chunks: int = 1
    output_actions: bool = True

    dreamzero_obs: dict[str, Any] | None = field(
        default=None, metadata={"batch_sig_exclude": True}
    )
