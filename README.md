# NeMo SpeechLM vLLM Inference

Documentation and orchestration instructions for running NeMo Speech LLM models
(NemotronH, Qwen3-based, Parakeet) with vLLM on the Draco cluster.

## Architecture

The vLLM plugin code lives in **NeMo** at
`nemo/collections/speechlm2/vllm/nemotron_v3/`.  It registers two model classes:

| Class | Backbone | Example |
|-------|----------|---------|
| `NeMoSpeechLMForConditionalGeneration` | Standard transformer | Qwen3, Parakeet-TDT |
| `NeMoSpeechLMHybridForConditionalGeneration` | Hybrid Mamba+MoE | NemotronH-30B |

Installing NeMo (`pip install -e NeMo`) registers the plugin via
`vllm.general_plugins` entry point.  No separate package installation is needed.

The evaluation pipeline scripts (`burst_eval_vllm.py`, `prepare_checkpoint_config.py`,
`serve_vllm_speechlm.sh`) live in
[canary-dev](https://gitlab-master.nvidia.com/pzelasko/canary-dev) under
`speechlm-2026h1/`.

### Related PRs

- NeMo PR #15520: https://github.com/NVIDIA-NeMo/NeMo/pull/15520 (plugin)
- Skills PR #1308: https://github.com/NVIDIA-NeMo/Skills/pull/1308 (eval backend)

## Cluster Job Submission

Evaluation jobs are submitted from the local machine using `burst_eval_vllm.py`,
which uses `nemo-run` to orchestrate SLURM jobs on Draco.

Cluster config lives at `canary-dev/speechlm-2026h1/cluster_configs/iad.yaml`
(created from `TEMPLATE.yaml`).  Key settings:

| Setting | Value |
|---------|-------|
| SSH host | `draco_oci_iad` |
| Job dir | `/lustre/fsw/portfolios/llmservice/users/dongjig/results/speechlm-2026h1` |
| Accounts | `convai_convaird_nemo-speech`, `nemotron_speech_asr`, `llmservice_nemo_speechlm` |
| Partitions | `batch_block1,batch_block2,batch_block3,batch_block4` |

Check queue load before submitting to pick the least busy account:
```bash
ssh draco_oci_iad "squeue -A convai_convaird_nemo-speech -h | wc -l"
ssh draco_oci_iad "squeue -A nemotron_speech_asr -h | wc -l"
```

## Prerequisites

- Access to Draco cluster (one of the accounts above)
- 1x A100-80GB GPU (per server instance)

## Container

Two options (use whichever works for your setup):

| Option | Path / URL |
|--------|-----------|
| **A) Squashfs (Draco cluster)** | `/lustre/fsw/portfolios/llmservice/users/dongjig/containers/vllm-asr-v3.sqsh` |
| **B) GitLab registry (any cluster)** | `gitlab-master.nvidia.com/dongjig/vllm-asr:v3` |

Option A is faster (pre-converted). Option B works on any cluster with Enroot/Pyxis (first pull is slow, cached after).

## Shared resources on Draco cluster

| Item | Path |
|------|------|
| Checkpoint | `/lustre/fsw/portfolios/llmservice/users/dongjig/models/nemotron-nano-asr-ckpt` |
| Test audio | `/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/asr_evaluator/datasets/HF-audio/open-asr-leaderboarddatasets-test-only-librispeech-test.clean/0_file_0.wav` |
| HF cache | `/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/users/dongjig/hf_cache` |

## Step 1: Clone the PR branches

On the login node (before starting the container, since git is not available inside):

```bash
WORKSPACE=/lustre/fsw/portfolios/llmservice/users/$USER/e2e_test_workspace
rm -rf ${WORKSPACE}
mkdir -p ${WORKSPACE}

# Clone NeMo PR #15520
git clone -b vllm-nemo-speechlm https://github.com/DongjiGao/NeMo.git ${WORKSPACE}/NeMo-PR

# Clone Skills PR #1308
git clone -b vllm-asr-backend https://github.com/DongjiGao/Skills.git ${WORKSPACE}/Skills-PR
```

## Step 2: Start a container session

**Option A** (squashfs on Draco):
```bash
srun -A llmservice_nemo_speechlm -p batch_singlenode \
    --gpus-per-node=1 -N 1 -t 0:30:00 --pty \
    --container-image=/lustre/fsw/portfolios/llmservice/users/dongjig/containers/vllm-asr-v3.sqsh \
    --container-mounts="/lustre:/lustre,/tmp:/tmp" \
    --container-workdir=/tmp \
    --no-container-mount-home \
    bash
```

**Option B** (GitLab registry, any cluster):
```bash
srun -A llmservice_nemo_speechlm -p batch_singlenode \
    --gpus-per-node=1 -N 1 -t 0:30:00 --pty \
    --container-image=gitlab-master.nvidia.com/dongjig/vllm-asr:v3 \
    --container-mounts="/lustre:/lustre,/tmp:/tmp" \
    --container-workdir=/tmp \
    --no-container-mount-home \
    bash
```
Note: Option B's first run takes ~5 min to pull and convert the image. Subsequent runs are cached.

## Step 3: Install the NeMo plugin (PR #15520)

Inside the container:

```bash
export HF_HUB_OFFLINE=1
export HF_HOME=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/users/dongjig/hf_cache
WORKSPACE=/lustre/fsw/portfolios/llmservice/users/$USER/e2e_test_workspace

# Install NeMo from the PR (--no-deps to avoid upgrading other packages)
pip install --no-deps -e ${WORKSPACE}/NeMo-PR
```

## Step 4: Test the NeMo plugin (PR #15520)

```bash
export VLLM_PLUGINS=nemo_speechlm

python3 -u -c "
import os, time, re
os.environ['VLLM_PLUGINS'] = 'nemo_speechlm'

from vllm import LLM, SamplingParams
import soundfile as sf

llm = LLM(
    model='/lustre/fsw/portfolios/llmservice/users/dongjig/models/nemotron-nano-asr-ckpt',
    tokenizer='nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16',
    hf_overrides={
        'architectures': ['NeMoSpeechLMHybridForConditionalGeneration'],
        'model_type': 'nemo_speechlm',
    },
    trust_remote_code=True,
    dtype='bfloat16',
    gpu_memory_utilization=0.90,
    enforce_eager=True,
    max_model_len=4096,
    limit_mm_per_prompt={'audio': 1},
)

tokenizer = llm.get_tokenizer()
messages = [{'role': 'user', 'content': 'Transcribe the following: <|audio|>'}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

audio, sr = sf.read('/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/asr_evaluator/datasets/HF-audio/open-asr-leaderboarddatasets-test-only-librispeech-test.clean/0_file_0.wav', dtype='float32')

outputs = llm.generate(
    [{'prompt': prompt, 'multi_modal_data': {'audio': (audio, sr)}}],
    SamplingParams(max_tokens=256, temperature=0.0),
)
text = re.sub(r'^<think>.*?</think>', '', outputs[0].outputs[0].text, flags=re.DOTALL).strip()
print('Transcription:', text)
"
```

**Expected output**: `Transcription: Concorde returned to its place amidst the tents,`

## Step 5: Install the Skills backend (PR #1308)

```bash
cp ${WORKSPACE}/Skills-PR/recipes/multimodal/server/backends/vllm_nemo_speechlm_backend.py \
   /opt/Skills/recipes/multimodal/server/backends/vllm_nemo_speechlm_backend.py
cp ${WORKSPACE}/Skills-PR/recipes/multimodal/server/backends/__init__.py \
   /opt/Skills/recipes/multimodal/server/backends/__init__.py
```

## Step 6: Test the Skills unified server (PR #1308)

Start the server:

```bash
export VLLM_PLUGINS=nemo_speechlm
python3 -m nemo_skills.inference.server.serve_unified \
    --backend vllm_nemo_speechlm \
    --model /lustre/fsw/portfolios/llmservice/users/dongjig/models/nemotron-nano-asr-ckpt \
    --tokenizer nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --gpu_memory_utilization 0.90 \
    --host 0.0.0.0 --port 8000 &

# Wait for server to be ready (look for "[Server] Ready!" in output)
sleep 420
```

Send a test request:

```bash
AUDIO_B64=$(base64 -w0 /lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/asr_evaluator/datasets/HF-audio/open-asr-leaderboarddatasets-test-only-librispeech-test.clean/0_file_0.wav)

curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{
        \"messages\": [{
            \"role\": \"user\",
            \"content\": [
                {\"type\": \"input_audio\", \"input_audio\": {\"data\": \"$AUDIO_B64\", \"format\": \"wav\"}},
                {\"type\": \"text\", \"text\": \"Transcribe the following:\"}
            ]
        }],
        \"max_tokens\": 512,
        \"temperature\": 0.0
    }" | python3 -m json.tool
```

**Expected**: JSON response with transcription in `choices[0].message.content`.

## Step 7: Run full evaluation on LibriSpeech (single GPU, offline mode)

This runs all 2,620 samples of librispeech_clean with batch_size=32 on a single GPU
and computes corpus-level WER. Uses vLLM's `LLM()` directly in-process (no HTTP server).
Good for quick validation and debugging.

```bash
export VLLM_PLUGINS=nemo_speechlm

python3 -u -c "
import json, os, time, re
os.environ['VLLM_PLUGINS'] = 'nemo_speechlm'

from vllm import LLM, SamplingParams
import soundfile as sf
import jiwer
from whisper_normalizer.english import EnglishTextNormalizer

MODEL = '/lustre/fsw/portfolios/llmservice/users/dongjig/models/nemotron-nano-asr-ckpt'
TOKENIZER = 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16'
DATA = '/lustre/fsw/portfolios/llmservice/users/dongjig/asr-leaderboard-data/nemo_skills_jsonl/librispeech_clean.jsonl'

llm = LLM(
    model=MODEL, tokenizer=TOKENIZER,
    hf_overrides={'architectures': ['NeMoSpeechLMHybridForConditionalGeneration'], 'model_type': 'nemo_speechlm'},
    trust_remote_code=True, dtype='bfloat16', gpu_memory_utilization=0.90,
    enforce_eager=True, max_model_len=4096, limit_mm_per_prompt={'audio': 1},
)

tokenizer = llm.get_tokenizer()
messages = [{'role': 'user', 'content': 'Transcribe the following: <|audio|>'}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

think_pat = re.compile(r'^<think>.*?</think>', re.DOTALL)
whisper_norm = EnglishTextNormalizer()
samples = [json.loads(l) for l in open(DATA)]

BATCH = 32
all_refs, all_hyps = [], []
t_start = time.time()

for i in range(0, len(samples), BATCH):
    batch = samples[i:i+BATCH]
    inputs, refs = [], []
    for s in batch:
        audio_path = s.get('audio_filepath', '')
        if not audio_path or not os.path.isfile(audio_path):
            continue
        audio, sr = sf.read(audio_path, dtype='float32')
        inputs.append({'prompt': prompt, 'multi_modal_data': {'audio': (audio, sr)}})
        refs.append(s.get('expected_answer', ''))
    if not inputs:
        continue
    outputs = llm.generate(inputs, SamplingParams(max_tokens=512, temperature=0.0), use_tqdm=False)
    for j, out in enumerate(outputs):
        hyp = think_pat.sub('', out.outputs[0].text).strip()
        r = whisper_norm(refs[j].lower())
        h = whisper_norm(hyp.lower())
        all_refs.append(r if r else 'empty')
        all_hyps.append(h if h else 'empty')
    if len(all_refs) % 512 == 0:
        print(f'  {len(all_refs)}/{len(samples)}')

wer = jiwer.wer(all_refs, all_hyps) * 100
print(f'Samples: {len(all_refs)}, WER: {wer:.2f}%, Time: {time.time()-t_start:.0f}s')
"
```

**Expected**: `WER: 1.91%` (matching NeMo checkpoint). Takes ~7.5 min on A100-80GB
(~99s model load from Lustre + ~349s inference).

## Step 8: Run evaluation with multi-GPU chunks (Skills pipeline)

For faster evaluation, use the NeMo-Skills `ns generate` pipeline which
orchestrates multi-GPU evaluation via Slurm. It splits data into N chunks,
launches N independent single-GPU servers (each with its own vLLM instance),
and auto-merges results when all chunks complete.

This tests the Skills PR #1308 backend integration.

**Differences from Step 7:**
- Step 7: single Python process, `LLM()` called directly, no server
- Step 8: Slurm-managed jobs, each chunk has a server + client, results auto-merged

**Why `serve_with_nvme.sh`?** The 60GB checkpoint loads ~5x faster from local
NVMe (`/tmp`) than from Lustre. The wrapper copies the checkpoint before starting
the server. This is especially important for multi-GPU jobs where all nodes
read the same Lustre file simultaneously, causing I/O contention.

**Prerequisites (one-time, on login node):**

```bash
pip install nemo-run hydra-core
```

**Run with 8 GPUs (8 independent servers):**

```bash
# Set these for your environment
CLUSTER_CONFIG=/home/dongjig/Skills/cluster_configs  # contains draco.yaml
DATA_DIR=/lustre/fsw/portfolios/llmservice/users/dongjig/asr-leaderboard-data/nemo_skills_jsonl
CKPT=/lustre/fsw/portfolios/llmservice/users/dongjig/models/nemotron-nano-asr-ckpt
OUTDIR=/lustre/fsw/portfolios/llmservice/users/$USER/results/librispeech_clean

ns generate \
    --cluster draco \
    --config_dir ${CLUSTER_CONFIG} \
    --input_file ${DATA_DIR}/librispeech_clean.jsonl \
    --output_dir ${OUTDIR} \
    --model ${CKPT} \
    --server_type generic \
    --server_entrypoint "bash /lustre/fsw/portfolios/llmservice/users/dongjig/scripts/serve_with_nvme.sh" \  # copies checkpoint to local NVMe before starting server
    --server_args "--backend vllm_nemo_speechlm --batch_size 32 \
        --tokenizer nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
        --gpu_memory_utilization 0.90" \
    --server_container vllm-asr \
    --server_gpus 1 \
    --num_chunks 8 \
    --expname librispeech-test \
    --installation_command "true" \
    ++prompt_format=openai ++prompt_config=null \
    ++enable_audio=true ++server.server_type=vllm_multimodal \
    ++max_concurrent_requests=32 \
    ++inference.temperature=0.0 ++inference.tokens_to_generate=512 \
    ++enable_audio_chunking=true \
    ++eval_type=audio ++eval_config.normalization_mode=hf_leaderboard
```

```

This submits 8 Slurm jobs. Each loads the model on 1 GPU, processes its chunk,
and writes `output_chunk_N.jsonl`. When all chunks finish, results are merged
into `output.jsonl`.

### Summarize results with `ns summarize_results`

After `ns generate` completes, compute corpus-level WER:

```bash
# Create directory structure expected by ns summarize_results
mkdir -p ${OUTDIR}/../summary/asr-leaderboard
ln -sf ${OUTDIR}/output.jsonl ${OUTDIR}/../summary/asr-leaderboard/output.jsonl

# Run summarization
ns summarize_results ${OUTDIR}/../summary
```

**Expected output**:
```
------------------------------------- asr-leaderboard -------------------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | wer   | num_entries
pass@1          | -1         | 108         | 99.54%       | 0.00%     | 1.91% | 2620
```

This computes corpus-level WER using the same method as the NeMo checkpoint evaluation.
Metrics are also saved to `metrics.json` for programmatic access.

**Monitor progress:**
```bash
squeue -u $USER  # check running jobs
ls /lustre/fsw/portfolios/llmservice/users/$USER/results/librispeech_clean/output_chunk_*.jsonl.done  # check completed chunks
```

**Expected**: ~1.91% WER.

Timing breakdown (8 chunks, librispeech_clean, A100-80GB):

| Phase | Per chunk | Notes |
|---|---|---|
| Server startup (NVMe copy + model load + warmup) | ~64s | Runs in parallel across all chunks |
| Inference (328 samples, BS=32) | ~77s | Slowest chunk |
| **Total wall-clock** | **~141s (2.4 min)** | vs 7.5 min single GPU (3.2x speedup) |

Speedup is larger on bigger datasets where inference dominates over model load.

**Available datasets** (all 8 Open ASR Leaderboard datasets):
```
librispeech_clean.jsonl   (2,620 samples)
librispeech_other.jsonl   (2,939 samples)
tedlium.jsonl             (1,155 samples)
spgispeech.jsonl          (39,341 samples)
voxpopuli.jsonl           (1,842 samples)
gigaspeech.jsonl          (19,898 samples)
earnings22.jsonl          (2,737 samples)
ami.jsonl                 (11,653 samples)
```
All at: `/lustre/fsw/portfolios/llmservice/users/dongjig/asr-leaderboard-data/nemo_skills_jsonl/`

## Step 9: Run full leaderboard with `ns eval` (all-in-one)

`ns eval` combines generate + evaluate + summarize in one command. It reads
benchmark data from `nemo_skills/dataset/asr-leaderboard/`, runs inference,
computes WER, and prints a summary table.

**Prerequisites**: The JSONL files in `nemo_skills/dataset/asr-leaderboard/`
must include audio paths in messages (see Step 1 for setup).

```bash
ns eval \
    --cluster draco \
    --config_dir /home/dongjig/Skills/cluster_configs \
    --benchmarks asr-leaderboard \
    --output_dir /lustre/fsw/portfolios/llmservice/users/$USER/results/ns_eval_test \
    --model /lustre/fsw/portfolios/llmservice/users/dongjig/models/nemotron-nano-asr-ckpt \
    --server_type generic \
    --server_entrypoint "bash /lustre/fsw/portfolios/llmservice/users/dongjig/scripts/serve_with_nvme.sh" \
    --server_args "--backend vllm_nemo_speechlm --batch_size 32 \
        --tokenizer nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
        --gpu_memory_utilization 0.90" \
    --server_container /lustre/fsw/portfolios/llmservice/users/dongjig/containers/vllm-asr-v3.sqsh \
    --server_gpus 1 \
    --num_chunks 8 \
    --expname ns-eval-test \
    --installation_command "true" \
    ++server.server_type=vllm_multimodal \
    ++max_concurrent_requests=32 \
    ++inference.temperature=0.0 \
    ++inference.tokens_to_generate=512 \
    ++enable_audio_chunking=true
```

After all chunks finish, run `ns summarize_results`:
```bash
ns summarize_results /lustre/fsw/portfolios/llmservice/users/$USER/results/ns_eval_test/eval-results
```

**Expected output** (verified):
```
------------------------------------- asr-leaderboard -------------------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | wer   | num_entries
pass@1          | -1         | 1309        | 98.43%       | 0.05%     | 4.65% | 67795

Dataset breakdown:
  librispeech_clean:  1.93%  (2,620 samples)
  librispeech_other:  3.69%  (2,939 samples)
  tedlium:            3.70%  (1,155 samples)
  spgispeech:         2.21%  (39,341 samples)
  voxpopuli:          6.27%  (1,842 samples)
  gigaspeech:        10.08%  (19,898 samples)
```

## Summary of expected results

| Test | Expected |
|------|----------|
| Step 4: Plugin inference | `Concorde returned to its place amidst the tents,` |
| Step 6: Server request | JSON with transcription |
| Step 7: Single GPU eval | 1.91% WER, ~7.5 min (99s load + 349s inference) |
| Step 8: 8-GPU eval (v3 container) | 1.91% WER, ~2.4 min wall-clock (46s load + 96s inference) |
| Step 9: Full leaderboard via ns eval | 4.65% avg WER across 6 datasets, ~20 min |
