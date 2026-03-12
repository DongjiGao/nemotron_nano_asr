def register():
    from nemotron_nano_asr.config import NemotronNanoASRConfig

    from transformers import AutoConfig
    AutoConfig.register("nemotron_nano_asr", NemotronNanoASRConfig)

    from vllm.transformers_utils.config import _CONFIG_REGISTRY
    _CONFIG_REGISTRY["nemotron_nano_asr"] = NemotronNanoASRConfig

    from vllm.model_executor.models.registry import ModelRegistry
    ModelRegistry.register_model(
        "NemotronNanoASRForConditionalGeneration",
        "nemotron_nano_asr.model:NemotronNanoASRForConditionalGeneration",
    )

    # Patch NemotronHConfig to add rms_norm_eps alias if missing
    # (vLLM 0.10 references it but HF config uses layer_norm_epsilon)
    try:
        from transformers import AutoConfig as _AC
        _nhc = _AC.from_pretrained(
            "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            trust_remote_code=True,
        )
        NHConfigCls = type(_nhc)
        _orig_getattr = getattr(NHConfigCls, '__getattr__', None)

        def _patched_getattr(self, name):
            if name == 'rms_norm_eps':
                return getattr(self, 'layer_norm_epsilon', 1e-5)
            if _orig_getattr:
                return _orig_getattr(self, name)
            raise AttributeError(name)

        NHConfigCls.__getattr__ = _patched_getattr
    except Exception:
        pass
