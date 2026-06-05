# MaCroScope-Tinker

Train a **General Reasoner**–style model on [Tinker](https://thinkingmachines.ai) with a single local GPU for the reward verifier. This repo contains exactly the scripts we use for MaCroScope Tinker runs on Alliance Canada (Fir).

## What this repo includes

| File | Purpose |
|------|---------|
| `scripts/tinker_train_general_reasoner.py` | Main GRPO/PPO training loop (Tinker cloud + local verifier) |
| `slurm/run_tinker_train.slurm` | Slurm launcher (1× H100 driver node) |
| `scripts/download_tinker_checkpoint.sh` | Download `sampler_weights/final` (or val checkpoints) from Tinker |
| `scripts/merge_lora_to_base.py` | Merge downloaded LoRA into the Nemotron base for vLLM eval |
| `scripts/push_to_hf.py` | Upload adapter or merged model to Hugging Face |
| `setup_env.sh` | One-time Python env + [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) install |

## Architecture

- **Training** (forward/backward, LoRA updates) runs on **Tinker** in the cloud.
- **Rollout verification** uses **vLLM** locally with `TIGER-Lab/general-verifier` on **1 GPU**.
- Checkpoints are stored on Tinker; `checkpoints.jsonl` in your log dir records `tinker://…` paths.

## Default hyperparameters

Configured in `scripts/tinker_train_general_reasoner.py` (`Config` class):

| Parameter | Default | Notes |
|-----------|---------|-------|
| Base model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16` | |
| Dataset | `ai4collaboration/MaCroScope-Biology-Chemistry-23k` | train ~20.7k / test ~2.3k |
| `total_epochs` | 5 | |
| `batch_size` | 90 | questions per step |
| `group_size` | 8 | rollouts per question |
| `learning_rate` | 5e-7 | Adam β1=0.9, β2=0.95 |
| `max_prompt_length` | 1024 | filters long prompts |
| `max_tokens` | 8192 | max generation length |
| `lora_rank` | 32 | |
| `save_freq` | 50 | state checkpoints |
| `test_freq` | 25 | eval on test split |

**Step count:** `floor(N_train / batch_size) × total_epochs`  
With ~20.7k train rows and batch 90 → **~1,150 steps** (fewer if many prompts exceed `max_prompt_length`).

Override at launch:

```bash
python scripts/tinker_train_general_reasoner.py \
    log_path=./log_my_run \
    total_epochs=3 \
    batch_size=100 \
    dataset_path=ai4collaboration/MaCroScope-Mathematics-10k
```

## Prerequisites

1. **Tinker API key** — sign up at [Tinker](https://thinkingmachines.ai) and get `TINKER_API_KEY`.
2. **1 local GPU** — hosts the verifier model during training (H100 recommended for Nemotron-scale verifier throughput).
3. **Python 3.11** with CUDA PyTorch, vLLM, and [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook).

Optional: `WANDB_API_KEY` for logging to project `MaCroScope-Tinker`.

## Environment setup

### Alliance Canada (Fir) — tested stack

```bash
module load python/3.11 cuda/12.2 gcc opencv arrow

# Clone this repo
git clone https://github.com/Anupam-Anupam/MaCroScope-Tinker.git
cd MaCroScope-Tinker

bash setup_env.sh
source ~/envs/macroscope/bin/activate
```

`setup_env.sh` creates `~/envs/macroscope`, installs `requirements.txt`, and editable-installs `tinker-cookbook` from `$SCRATCH/tinker-cookbook`.

### Secrets

```bash
cp .env.example ~/.tinker_env
chmod 600 ~/.tinker_env
# Edit: TINKER_API_KEY, optionally WANDB_API_KEY and HF_TOKEN
```

### Other clusters

1. Install CUDA-matched PyTorch from [pytorch.org](https://pytorch.org).
2. `pip install -r requirements.txt`
3. `git clone https://github.com/thinking-machines-lab/tinker-cookbook.git && pip install -e tinker-cookbook`

## Training

### Interactive (login / dev node with GPU)

```bash
source ~/envs/macroscope/bin/activate
source ~/.tinker_env

cd MaCroScope-Tinker
python scripts/tinker_train_general_reasoner.py \
    log_path=./log_nemotron_biochem23k_grstyle
```

### Slurm (Fir)

```bash
cd MaCroScope-Tinker
sbatch slurm/run_tinker_train.slurm
```

Logs: `tinker-nemotron-biochem23k-<jobid>.{out,err}` in the submit directory.  
Training metadata: `./log_nemotron_biochem23k_grstyle/checkpoints.jsonl`.

## After training

### 1. Download final LoRA from Tinker

Find your run id in `log_*/checkpoints.jsonl` (e.g. `1591a9f4-…:train:0`).

```bash
export TINKER_RUN_ID="YOUR_RUN_ID:train:0"
bash scripts/download_tinker_checkpoint.sh final
```

Adapter lands in `./checkpoints/<run_id>_sampler_weights_final/` with `adapter_model.safetensors`.

### 2. Merge LoRA into base (for local vLLM eval)

Nemotron uses custom code; you need `mamba-ssm` for merge. On Alliance:

```bash
pip install causal-conv1d
pip install mamba-ssm==2.2.4   # use CC wheelhouse if available
```

```bash
python scripts/merge_lora_to_base.py \
    --base nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 \
    --adapter ./checkpoints/YOUR_RUN_sampler_weights_final \
    --out ./merged-final \
    --dtype bfloat16 \
    --device-map cpu
```

Point `HF_HOME` to scratch if home quota is tight:

```bash
export HF_HOME="$SCRATCH/huggingface_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
```

### 3. Push to Hugging Face

```bash
export HF_TOKEN="hf_..."
python scripts/push_to_hf.py \
    --local-dir ./checkpoints/YOUR_RUN_sampler_weights_final \
    --repo-id your-org/your-lora-repo \
    --commit-message "Final Tinker LoRA"
```

## Hugging Face assets

| Asset | Repo |
|-------|------|
| Training dataset (default) | [ai4collaboration/MaCroScope-Biology-Chemistry-23k](https://huggingface.co/datasets/ai4collaboration/MaCroScope-Biology-Chemistry-23k) |
| Base model | [nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16) |
| Verifier | [TIGER-Lab/general-verifier](https://huggingface.co/TIGER-Lab/general-verifier) |

## Acknowledgements

- Training loop adapted from [tinker-cookbook `rl_loop.py`](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/recipes/rl_loop.py)
- General Reasoner: [Ma et al., NeurIPS 2025](https://openreview.net/forum?id=pBFVoll8Xa)

## Citation

```bibtex
@inproceedings{ma2025generalreasoner,
  title={General-Reasoner: Advancing {LLM} Reasoning Across All Domains},
  author={Xueguang Ma and Qian Liu and Dongfu Jiang and Ge Zhang and Zejun MA and Wenhu Chen},
  booktitle={NeurIPS},
  year={2025}
}
```
