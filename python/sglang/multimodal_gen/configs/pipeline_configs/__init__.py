# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

__all__ = [
    "DiffusersGenericPipelineConfig",
    "HeliosDistilledConfig",
    "HeliosMidConfig",
    "HeliosT2VConfig",
    "HunyuanConfig",
    "FastHunyuanConfig",
    "Hunyuan3D2PipelineConfig",
    "FluxPipelineConfig",
    "Flux2PipelineConfig",
    "Flux2KleinPipelineConfig",
    "Flux2FinetunedPipelineConfig",
    "PipelineConfig",
    "SanaPipelineConfig",
    "SlidingTileAttnConfig",
    "MOVAPipelineConfig",
    "StableDiffusion3PipelineConfig",
    "WanT2V480PConfig",
    "WanI2V480PConfig",
    "WanT2V720PConfig",
    "WanI2V720PConfig",
    "SelfForcingWanT2V480PConfig",
    "ZImagePipelineConfig",
    "LTX2PipelineConfig",
    "DreamZeroPipelineConfig",
]


def __getattr__(name):
    if name in {"PipelineConfig", "SlidingTileAttnConfig"}:
        from sglang.multimodal_gen.configs.pipeline_configs.base import (
            PipelineConfig,
            SlidingTileAttnConfig,
        )

        return {
            "PipelineConfig": PipelineConfig,
            "SlidingTileAttnConfig": SlidingTileAttnConfig,
        }[name]
    if name == "DiffusersGenericPipelineConfig":
        from sglang.multimodal_gen.configs.pipeline_configs.diffusers_generic import (
            DiffusersGenericPipelineConfig,
        )

        return DiffusersGenericPipelineConfig
    if name in {"FluxPipelineConfig", "Flux2PipelineConfig", "Flux2KleinPipelineConfig"}:
        from sglang.multimodal_gen.configs.pipeline_configs.flux import (
            Flux2KleinPipelineConfig,
            Flux2PipelineConfig,
            FluxPipelineConfig,
        )

        return {
            "FluxPipelineConfig": FluxPipelineConfig,
            "Flux2PipelineConfig": Flux2PipelineConfig,
            "Flux2KleinPipelineConfig": Flux2KleinPipelineConfig,
        }[name]
    if name == "Flux2FinetunedPipelineConfig":
        from sglang.multimodal_gen.configs.pipeline_configs.flux_finetuned import (
            Flux2FinetunedPipelineConfig,
        )

        return Flux2FinetunedPipelineConfig
    if name in {"HeliosDistilledConfig", "HeliosMidConfig", "HeliosT2VConfig"}:
        from sglang.multimodal_gen.configs.pipeline_configs.helios import (
            HeliosDistilledConfig,
            HeliosMidConfig,
            HeliosT2VConfig,
        )

        return {
            "HeliosDistilledConfig": HeliosDistilledConfig,
            "HeliosMidConfig": HeliosMidConfig,
            "HeliosT2VConfig": HeliosT2VConfig,
        }[name]
    if name in {"HunyuanConfig", "FastHunyuanConfig"}:
        from sglang.multimodal_gen.configs.pipeline_configs.hunyuan import (
            FastHunyuanConfig,
            HunyuanConfig,
        )

        return {
            "HunyuanConfig": HunyuanConfig,
            "FastHunyuanConfig": FastHunyuanConfig,
        }[name]
    if name == "Hunyuan3D2PipelineConfig":
        from sglang.multimodal_gen.configs.pipeline_configs.hunyuan3d import (
            Hunyuan3D2PipelineConfig,
        )

        return Hunyuan3D2PipelineConfig
    if name == "LTX2PipelineConfig":
        from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import (
            LTX2PipelineConfig,
        )

        return LTX2PipelineConfig
    if name == "MOVAPipelineConfig":
        from sglang.multimodal_gen.configs.pipeline_configs.mova import MOVAPipelineConfig

        return MOVAPipelineConfig
    if name == "SanaPipelineConfig":
        from sglang.multimodal_gen.configs.pipeline_configs.sana import SanaPipelineConfig

        return SanaPipelineConfig
    if name == "StableDiffusion3PipelineConfig":
        from sglang.multimodal_gen.configs.pipeline_configs.stablediffusion3 import (
            StableDiffusion3PipelineConfig,
        )

        return StableDiffusion3PipelineConfig
    if name in {
        "SelfForcingWanT2V480PConfig",
        "WanI2V480PConfig",
        "WanI2V720PConfig",
        "WanT2V480PConfig",
        "WanT2V720PConfig",
    }:
        from sglang.multimodal_gen.configs.pipeline_configs.wan import (
            SelfForcingWanT2V480PConfig,
            WanI2V480PConfig,
            WanI2V720PConfig,
            WanT2V480PConfig,
            WanT2V720PConfig,
        )

        return {
            "SelfForcingWanT2V480PConfig": SelfForcingWanT2V480PConfig,
            "WanI2V480PConfig": WanI2V480PConfig,
            "WanI2V720PConfig": WanI2V720PConfig,
            "WanT2V480PConfig": WanT2V480PConfig,
            "WanT2V720PConfig": WanT2V720PConfig,
        }[name]
    if name == "ZImagePipelineConfig":
        from sglang.multimodal_gen.configs.pipeline_configs.zimage import (
            ZImagePipelineConfig,
        )

        return ZImagePipelineConfig
    if name == "DreamZeroPipelineConfig":
        from sglang.multimodal_gen.configs.pipeline_configs.dreamzero import (
            DreamZeroPipelineConfig,
        )

        return DreamZeroPipelineConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
