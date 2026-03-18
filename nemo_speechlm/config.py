"""Configuration for NeMo Speech LM models in vLLM.

Supports any combination of NeMo speech encoder + LLM backbone.
The checkpoint config.json defines which components to use.
"""

from transformers import AutoConfig, PretrainedConfig


class NeMoSpeechLMConfig(PretrainedConfig):
    model_type = "nemo_speechlm"

    def __init__(
        self,
        perception: dict | None = None,
        pretrained_llm: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        pretrained_asr: str = "nvidia/canary-1b-v2",
        audio_locator_tag: str = "<|audio|>",
        prompt_format: str = "nemotron-nano-v3",
        pretrained_weights: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.perception = perception or {}
        self.pretrained_llm = pretrained_llm
        self.pretrained_asr = pretrained_asr
        self.audio_locator_tag = audio_locator_tag
        self.prompt_format = prompt_format
        self.pretrained_weights = pretrained_weights

        self.text_config = AutoConfig.from_pretrained(
            pretrained_llm, trust_remote_code=True
        )
        self.text_config.architectures = ["NemotronHForCausalLM"]

        if not hasattr(self.text_config, "total_num_kv_heads") or \
                self.text_config.total_num_kv_heads is None:
            self.text_config.total_num_kv_heads = getattr(
                self.text_config, "num_key_value_heads", 2
            )

        if not hasattr(self.text_config, "rms_norm_eps"):
            self.text_config.rms_norm_eps = getattr(
                self.text_config, "layer_norm_epsilon", 1e-5
            )

        self.text_config.vocab_size = self.text_config.vocab_size + 10

    def get_text_config(self, decoder=False) -> PretrainedConfig:
        return self.text_config

    _ATTR_ALIASES = {
        "rms_norm_eps": "layer_norm_epsilon",
        "layer_norm_eps": "layer_norm_epsilon",
    }

    def __getattr__(self, name):
        if name.startswith("_") or name in (
            "perception", "pretrained_llm", "pretrained_asr",
            "audio_locator_tag", "prompt_format", "pretrained_weights",
            "text_config", "_ATTR_ALIASES",
        ):
            raise AttributeError(name)
        alias = self._ATTR_ALIASES.get(name, name)
        try:
            return getattr(self.text_config, alias)
        except AttributeError:
            if alias != name:
                try:
                    return getattr(self.text_config, name)
                except AttributeError:
                    pass
            raise AttributeError(
                f"'{type(self).__name__}' has no attribute '{name}'"
            )
