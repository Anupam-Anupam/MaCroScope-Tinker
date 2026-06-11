# Fir → Clariden migration

You **cannot copy** the Fir `~/envs/macroscope` venv to Clariden:

| Fir | Clariden |
|-----|----------|
| x86_64 (H100) | **ARM64** (GH200) |
| Alliance CVMFS wheels (`+computecanada`) | NGC containers / uenv / conda aarch64 |
| `/scratch/anupam` | `/iopsstor/scratch/cscs/$USER` |
| `module load python/3.11 cuda/12.2` | EDF container or `prgenv-gnu` uenv |

**What transfers:** code, dependency list, secrets (manually), Tinker cloud checkpoints (already on Tinker).

---

## Step 1 — On Fir: create transport bundle

```bash
cd /scratch/anupam/MaCroScope-Tinker   # or clone from GitHub
bash clariden/export_from_fir.sh
```

Creates `$SCRATCH/clariden_transport_bundle.tar.gz` containing:

- `MaCroScope-Tinker/` (this repo)
- `tinker-cookbook/` (if present on Fir scratch)
- `requirements-fir-freeze.txt` (reference)
- `user.env.example` (no secrets)

**Copy secrets separately** (never put in tarball):

```bash
# On your laptop or secure channel — copy ~/.tinker_env from Fir to Clariden as ~/user.env
scp fir:~/.tinker_env clariden:~/user.env
chmod 600 ~/user.env
```

---

## Step 2 — Transfer bundle to Clariden

From a machine that can reach both (or laptop as relay):

```bash
scp /scratch/anupam/clariden_transport_bundle.tar.gz clariden:~/scratch/
```

Or use [Swiss AI `cscs-cl`](https://github.com/swiss-ai/reasoning_getting-started) SSH setup.

---

## Step 3 — On Clariden: unpack

```bash
mkdir -p ~/scratch
tar -xzf ~/scratch/clariden_transport_bundle.tar.gz -C ~/scratch/
cd ~/scratch/clariden_transport_bundle/MaCroScope-Tinker
```

**Alternative (recommended):** skip tarball, clone fresh:

```bash
cd ~/scratch
git clone https://github.com/Anupam-Anupam/MaCroScope-Tinker.git
cd MaCroScope-Tinker
```

---

## Step 4 — Clariden environment (container + venv)

Follow [Swiss AI reasoning_getting-started](https://github.com/swiss-ai/reasoning_getting-started) for:

1. `cscs-cl` SSH access
2. EDF environment `my_env` with NGC PyTorch ARM image, e.g. `nvcr.io/nvidia/pytorch:25.01-py3`
3. Optional miniconda **aarch64**: `Miniconda3-latest-Linux-aarch64.sh`

Inside an interactive container session (`sdebug --environment=my_env bash`):

```bash
cd ~/scratch/MaCroScope-Tinker
bash clariden/setup_clariden.sh
source ~/envs/macroscope/bin/activate
python -c "import torch, tinker, vllm; print(torch.cuda.get_device_name(), 'ok')"
```

If `vllm` fails on ARM, check CSCS docs for supported vLLM builds or use a pre-built NGC image with vLLM.

---

## Step 5 — Secrets and caches

```bash
cp clariden/user.env.example ~/user.env
chmod 600 ~/user.env
# Edit: TINKER_API_KEY, WANDB_API_KEY, HF_TOKEN
```

Caches go on **iopsstor** (not 50GB `/users`):

```bash
export HF_HOME=/iopsstor/scratch/cscs/$USER/huggingface_cache
```

---

## Step 6 — Submit training job

Edit `clariden/run_tinker_train_clariden.slurm`:

- `YOUR_PROJECT_ACCOUNT` → your CSCS project account
- `--environment=my_env` → your EDF environment name

```bash
sbatch clariden/run_tinker_train_clariden.slurm
```

Logs: `/iopsstor/scratch/cscs/$USER/tinker-biochem23k-<jobid>.out`

---

## What stays on Tinker (no migration needed)

Training weights and run state live on **Tinker cloud**. After switching clusters, download checkpoints with:

```bash
export TINKER_RUN_ID="your-run-id:train:0"
bash scripts/download_tinker_checkpoint.sh final
```

---

## Fir vs Clariden quick reference

| Item | Fir | Clariden |
|------|-----|----------|
| Repo | `/scratch/anupam/MaCroScope-Tinker` | `~/scratch/MaCroScope-Tinker` |
| Env | `source ~/envs/macroscope/bin/activate` | same path after `setup_clariden.sh` |
| tinker-cookbook | `$SCRATCH/tinker-cookbook` | `~/scratch/tinker-cookbook` |
| Slurm script | `slurm/run_tinker_train.slurm` | `clariden/run_tinker_train_clariden.slurm` |
| HF cache | `$SCRATCH/huggingface_cache` | `/iopsstor/scratch/cscs/$USER/huggingface_cache` |
