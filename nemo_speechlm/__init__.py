"""vLLM plugin registration for NeMo Speech LM models (standalone package)."""

_PKG = "nemo_speechlm"


def register():
    from nemo_speechlm.config import NeMoSpeechLMConfig

    from transformers import AutoConfig
    AutoConfig.register("nemo_speechlm", NeMoSpeechLMConfig)

    from vllm.transformers_utils.config import _CONFIG_REGISTRY
    _CONFIG_REGISTRY["nemo_speechlm"] = NeMoSpeechLMConfig

    from vllm.model_executor.models.registry import ModelRegistry
    ModelRegistry.register_model(
        "NeMoSpeechLMForConditionalGeneration",
        f"{_PKG}.model:NeMoSpeechLMForConditionalGeneration",
    )
    ModelRegistry.register_model(
        "NeMoSpeechLMStdForConditionalGeneration",
        f"{_PKG}.model:NeMoSpeechLMStdForConditionalGeneration",
    )

    _apply_backend_patches()


def _apply_backend_patches():
    try:
        from transformers import AutoConfig as _AC
        _nhc = _AC.from_pretrained(
            "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            trust_remote_code=True,
        )
        NHConfigCls = type(_nhc)
        _orig_getattr = getattr(NHConfigCls, "__getattr__", None)

        def _patched_getattr(self, name):
            if name == "rms_norm_eps":
                return getattr(self, "layer_norm_epsilon", 1e-5)
            if _orig_getattr:
                return _orig_getattr(self, name)
            raise AttributeError(name)

        NHConfigCls.__getattr__ = _patched_getattr
    except Exception:
        pass
