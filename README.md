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

## Step 8: Run cluster evaluation with `burst_eval_vllm.py`

The recommended way to run full leaderboard evaluations on the cluster.
The script lives in [canary-dev](https://gitlab-master.nvidia.com/dongjig/canary-dev)
at `speechlm-2026h1/burst_eval_vllm.py`.

It handles everything automatically:
- **Auto-detects** model architecture (hybrid vs standard) and tokenizer from `config.json`
- **Uploads** NeMo code + server wrapper to the cluster
- **Submits** N parallel SLURM jobs (each with its own vLLM server + eval client)
- **Merges** results and computes metrics

### Prerequisites (one-time, on local machine)

```bash
# Install nemo-skills (for pipeline orchestration)
cd ~/Skills && pip install -e . --no-deps

# Create cluster config from template
cd ~/canary-dev/speechlm-2026h1
cp cluster_configs/TEMPLATE.yaml cluster_configs/iad.yaml
# Edit iad.yaml: set ssh_tunnel.host, user, job_dir, account
```

### Prepare a checkpoint for vLLM (one-time per checkpoint)

Before evaluating, patch the checkpoint's config for vLLM:

```bash
cd ~/canary-dev/speechlm-2026h1

python prepare_checkpoint_config.py /path/to/checkpoint

# Dry-run to see what would change:
python prepare_checkpoint_config.py /path/to/checkpoint --dry-run
```

This patches `config.json` (model_type, architectures), copies the tokenizer
from the base LLM, overrides the chat template, and writes `generation_config.json`
for greedy decoding.

### Evaluate a single pre-converted checkpoint

```bash
cd ~/canary-dev/speechlm-2026h1

PYTHONPATH=~/Skills:$PYTHONPATH \
NEMO_SKILLS_CONFIG=$PWD/cluster_configs \
NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1 \
python burst_eval_vllm.py \
    --model-dir /lustre/fsw/.../canary-qwen-2.5b-hf \
    --cluster iad \
    --benchmarks asr-leaderboard \
    --server-gpus 1 \
    --num-chunks 8 \
    --nemo-dir /tmp/NeMo_lite \
    --data-dir /lustre/fsw/.../skills_data
```

Key arguments:

| Argument | Description |
|----------|-------------|
| `--model-dir` | Path to the HF checkpoint on the cluster |
| `--cluster` | Cluster config name (matches `cluster_configs/iad.yaml`) |
| `--benchmarks` | `asr-leaderboard`, `audio`, `text`, `all`, or comma-separated list |
| `--server-gpus` | GPUs per vLLM server instance (1 for most models) |
| `--num-chunks` | Number of parallel eval chunks (each gets its own server) |
| `--nemo-dir` | Local NeMo directory to upload (must contain the vLLM plugin) |
| `--data-dir` | Path to prepared Skills dataset on the cluster |
| `--tokenizer` | Auto-detected from checkpoint config; override if needed |
| `--hf-overrides` | Auto-detected architecture; override if needed |
| `--dry-run` | Print commands without submitting |

### Burst mode (evaluate all checkpoints in an experiment)

```bash
python burst_eval_vllm.py \
    --expname nano-v3-canary-v2-4node \
    --config nano-v3-canary-v2-asr.yaml \
    --cluster iad \
    --benchmarks asr-leaderboard \
    --server-gpus 1 \
    --num-chunks 8 \
    --nemo-dir /tmp/NeMo_lite \
    --data-dir /lustre/fsw/.../skills_data
```

This discovers all `*.ckpt` files, converts them to HF format, and evaluates each.

### Monitor and collect results

```bash
# Check job status
ssh draco_oci_iad "squeue -u dongjig"

# After eval chunks complete, cancel the idle server job to unblock metrics
ssh draco_oci_iad "scancel <server_job_id>"

# Compute metrics manually if the metrics job didn't run
PYTHONPATH=~/Skills:$PYTHONPATH \
NEMO_SKILLS_CONFIG=$PWD/cluster_configs \
python -m nemo_skills.pipeline.summarize_results \
    /lustre/fsw/.../vllm_eval/eval-results \
    --benchmarks asr-leaderboard \
    --cluster iad
```

### Verified results (canary-qwen-2.5b, ASR leaderboard)

| Dataset | WER |
|---------|-----|
| **Average** | **5.56%** |
| librispeech_clean | 1.62% |
| librispeech_other | 3.07% |
| tedlium | 2.64% |
| spgispeech | 1.90% |
| voxpopuli | 5.60% |
| gigaspeech | 9.23% |
| earnings22 | 10.44% |
| ami | 9.99% |

### How it works internally

1. `burst_eval_vllm.py` reads the checkpoint's `config.json` via SSH
2. Auto-detects the LLM backbone and selects the correct vLLM model class
3. Uploads NeMo code + `serve_vllm_speechlm.sh` to the cluster
4. Submits N SLURM jobs via `nemo-run`, each running:
   - `serve_vllm_speechlm.sh`: installs NeMo plugin, exports `VLLM_PLUGINS=nemo_speechlm`, starts vLLM server
   - Eval client: sends audio requests to the server, writes `output_chunk_N.jsonl`
5. When the client finishes, the job kills the server and exits
6. A final metrics job merges chunks and computes WER

### Available benchmark datasets

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

## Summary of expected results

| Test | Expected |
|------|----------|
| Step 4: Plugin inference | `Concorde returned to its place amidst the tents,` |
| Step 6: Server request | JSON with transcription |
| Step 7: Single GPU eval (NemotronH) | 1.91% WER on librispeech_clean |
| Step 8: Cluster eval (canary-qwen-2.5b) | 5.56% avg WER across 8 ASR leaderboard datasets |
