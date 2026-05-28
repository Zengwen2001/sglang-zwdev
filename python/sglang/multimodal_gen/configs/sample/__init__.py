# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

__all__ = [
    "SamplingParams",
    "DiffusersGenericSamplingParams",
    "DreamZeroSamplingParams",
]


def __getattr__(name):
    if name == "SamplingParams":
        from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams

        return SamplingParams
    if name == "DiffusersGenericSamplingParams":
        from sglang.multimodal_gen.configs.sample.diffusers_generic import (
            DiffusersGenericSamplingParams,
        )

        return DiffusersGenericSamplingParams
    if name == "DreamZeroSamplingParams":
        from sglang.multimodal_gen.configs.sample.dreamzero import (
            DreamZeroSamplingParams,
        )

        return DreamZeroSamplingParams
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
