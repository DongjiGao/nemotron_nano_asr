def register():
    from nemotron_nano_asr.config import NemotronNanoASRConfig

    from vllm.transformers_utils.config import _CONFIG_REGISTRY
    _CONFIG_REGISTRY["nemotron_nano_asr"] = NemotronNanoASRConfig

    from vllm.model_executor.models.registry import ModelRegistry
    ModelRegistry.register_model(
        "NemotronNanoASRForConditionalGeneration",
        "nemotron_nano_asr.model:NemotronNanoASRForConditionalGeneration",
    )
