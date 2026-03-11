"""Configuration for Nemotron-Nano-v3 + Canary-v2 ASR model.

The checkpoint uses NeMo format config.json. Users must pass hf_overrides:
    --hf-overrides '{"architectures": ["NemotronNanoASRForConditionalGeneration"],
                     "model_type": "nemotron_nano_asr"}'
"""

from transformers import AutoConfig, PretrainedConfig


class NemotronNanoASRConfig(PretrainedConfig):
    model_type = "nemotron_nano_asr"

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

        # Extend vocab to accommodate audio special tokens that will be
        # added to the tokenizer at runtime. The embedding layer uses
        # org_num_embeddings for weight loading so the checkpoint stays
        # compatible.
        self.text_config.vocab_size = self.text_config.vocab_size + 10

    def get_text_config(self, decoder=False) -> PretrainedConfig:
        return self.text_config

    def __getattr__(self, name):
        if name.startswith("_") or name in (
            "perception", "pretrained_llm", "pretrained_asr",
            "audio_locator_tag", "prompt_format", "pretrained_weights",
            "text_config",
        ):
            raise AttributeError(name)
        try:
            return getattr(self.text_config, name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' has no attribute '{name}'"
            )
