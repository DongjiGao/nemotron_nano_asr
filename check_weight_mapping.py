"""Pre-check: compare checkpoint weight names vs model parameter names."""

import struct
import json


def get_checkpoint_keys(path):
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
    return {
        k: v["shape"]
        for k, v in header.items()
        if k != "__metadata__"
    }


def nemo_to_hf(name):
    if "._extra_state" in name:
        return None
    if not name.startswith("llm."):
        return None

    hf = name.replace("llm.model.", "backbone.")
    hf = hf.replace("llm.lm_head", "lm_head")

    if hf == "backbone.norm.weight":
        hf = "backbone.norm_f.weight"

    if hf.endswith(".experts.down_projs"):
        return hf.replace(".experts.down_projs", ".experts.{i}.down_proj.weight")
    if hf.endswith(".experts.gate_and_up_projs"):
        return hf.replace(".experts.gate_and_up_projs", ".experts.{i}.up_proj.weight")

    return hf


def get_model_params():
    import os
    os.environ["HF_HOME"] = "/home/dongjig/.cache/hf_home"
    os.environ["VLLM_PLUGINS"] = "nemotron_nano_asr"

    import torch
    from nemotron_nano_asr.config import NemotronNanoASRConfig
    from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM

    config = NemotronNanoASRConfig()
    hf_to_vllm = {
        "backbone": "model",
        "A_log": "A",
        "embeddings": "embed_tokens",
    }

    # Get expected param names from vLLM's NemotronH
    # We need to instantiate but on meta device
    from vllm.config import VllmConfig, ModelConfig
    from unittest.mock import MagicMock

    # Just get param names from the HF model definition
    from transformers import AutoConfig
    text_config = AutoConfig.from_pretrained(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        trust_remote_code=True,
    )

    # Instead of instantiating the full model, just check what the
    # hf_to_vllm_mapper would produce and what params_dict expects
    print("\n=== hf_to_vllm_mapper ===")
    print("backbone.* -> model.*")
    print("A_log -> A")
    print("embeddings -> embed_tokens")

    return text_config


def main():
    ckpt = "/home/dongjig/nemotron-nano-asr-ckpt/model.safetensors"
    ckpt_keys = get_checkpoint_keys(ckpt)

    print(f"Checkpoint has {len(ckpt_keys)} keys")

    llm_keys = {k: v for k, v in ckpt_keys.items() if k.startswith("llm.")}
    perception_keys = {k: v for k, v in ckpt_keys.items() if k.startswith("perception.")}
    extra_keys = {k: v for k, v in ckpt_keys.items() if "._extra_state" in k}

    print(f"  LLM keys: {len(llm_keys)}")
    print(f"  Perception keys: {len(perception_keys)}")
    print(f"  Extra state keys (skipped): {len(extra_keys)}")

    print(f"\n=== NeMo -> HF name conversion ===")
    converted = {}
    skipped = []
    expanded_experts = 0

    for name, shape in llm_keys.items():
        hf = nemo_to_hf(name)
        if hf is None:
            skipped.append(name)
            continue
        if "{i}" in hf:
            n_experts = shape[0]
            expanded_experts += n_experts
            for i in range(min(3, n_experts)):
                print(f"  {name} {shape} -> {hf.replace('{i}', str(i))} (x{n_experts})")
            if n_experts > 3:
                print(f"    ... ({n_experts} experts total)")
            converted[hf] = shape
        else:
            converted[hf] = shape

    print(f"\n  Converted: {len(converted)} unique patterns")
    print(f"  Expanded experts: {expanded_experts} individual weights")
    print(f"  Skipped: {len(skipped)} keys")
    if skipped:
        for s in skipped[:5]:
            print(f"    {s}")

    # Apply hf_to_vllm_mapper: backbone -> model, A_log -> A, embeddings -> embed_tokens
    print(f"\n=== After hf_to_vllm_mapper (backbone->model, A_log->A) ===")
    vllm_names = {}
    for hf, shape in converted.items():
        vllm = hf.replace("backbone.", "model.")
        vllm = vllm.replace("A_log", "A")
        vllm = vllm.replace("embeddings", "embed_tokens")
        vllm_names[vllm] = shape

    # Show a sample of final vLLM names
    samples = sorted(vllm_names.keys())
    for s in samples:
        if any(x in s for x in ["layers.0.", "layers.1.", "lm_head", "embed_tokens", "norm_f"]):
            if "{i}" not in s:
                print(f"  {s} {vllm_names[s]}")

    # Show MoE pattern
    moe_patterns = [s for s in samples if "{i}" in s]
    if moe_patterns:
        print(f"\n  MoE patterns ({len(moe_patterns)}):")
        for p in moe_patterns[:4]:
            print(f"    {p} {vllm_names[p]}")


if __name__ == "__main__":
    main()
