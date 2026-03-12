"""Quick test: Nemotron-Nano-v3 + Canary-v2 ASR via vLLM plugin."""

import numpy as np
import datasets
from vllm import LLM, SamplingParams


def main():
    ds = datasets.load_dataset(
        "hf-audio/esb-datasets-test-only-sorted",
        "librispeech",
        split="test.clean",
    )

    hf_overrides = {
        "architectures": ["NemotronNanoASRForConditionalGeneration"],
        "model_type": "nemotron_nano_asr",
    }

    print("Loading Nemotron-Nano ASR via vLLM plugin...")
    llm = LLM(
        model="/home/dongjig/nemotron-nano-asr-ckpt",
        tokenizer="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        hf_overrides=hf_overrides,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        max_model_len=4096,
        block_size=64,
        limit_mm_per_prompt={"audio": 1},
    )
    sampling_params = SamplingParams(max_tokens=256, temperature=0.0)

    prompt = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nTranscribe the following: <|audio|><|im_end|>\n<|im_start|>assistant\n<think>\n"

    for idx in [0, 9]:
        item = ds[idx]
        audio_arr = item["audio"]["array"].astype(np.float32)
        sr = item["audio"]["sampling_rate"]
        ref = item.get("text", "")[:80]

        print(f"\n{'='*60}")
        print(f"SAMPLE {idx}: len={len(audio_arr)} ({len(audio_arr)/sr:.1f}s)")
        print(f"  ref: {ref}...")

        outputs = llm.generate(
            {"prompt": prompt, "multi_modal_data": {"audio": (audio_arr, sr)}},
            sampling_params,
            use_tqdm=False,
        )
        hyp = outputs[0].outputs[0].text
        print(f"  hyp: {hyp[:100]}")


if __name__ == "__main__":
    main()
