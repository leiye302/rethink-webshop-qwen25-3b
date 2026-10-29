# RETHINK WebShop — Qwen2.5-3B

Route-SFT followed by **public SDAR GRPO+OPSD (gate beta 5)**, with extra
Teacher-only local hints at decisions where the **Actor itself generates
`<RETHINK>`**. This repository starts stage 2 from the released **1500-step
WebShop Route-SFT checkpoint**, not from SEED/M1 or an Analyzer checkpoint.

The [1.5B companion](https://github.com/leiye302/rethink-webshop-qwen25-1.5b)
starts from its public base and trains the same frozen SFT data first.
Neither repository needs the authors' computer, servers, private datasets or
preinstalled environments. Model weights are separate downloadable assets.

## What is fixed

| Component | Recipe |
|---|---|
| Base | Qwen2.5-3B-Instruct, pinned revision in `configs/model.json` |
| SFT data | 7,500 natural Good + 4,500 reviewed Bad; identical in both repositories |
| SFT | 1 epoch / 1,500 updates; every batch 5 Good + 3 Bad; full parameters |
| SFT loss | Sample-mean route/action/EOS CE + 0.1 independent full-vocabulary marker BCE; **no reference KL** |
| Stage 2 | 150 GRPO+OPSD updates; beta=5, distillation coefficient=0.01 |
| Teacher | Current Actor weights, no-gradient scoring view; **not a frozen initial Teacher** |
| Hints | Static public SDAR WebShop skills everywhere; additional local API hint only at Actor-marker decisions |
| Environment | SDAR default 1,000 products + synthetic goals, history 2, at most 15 actions |
| Budgets | Prompt 4,096 / response 512; unchanged prompt/action parser/reward |
| RL validation | Same fixed reserved 128 goals for both arms/sizes, every 10 updates |
| RL checkpoints | Every 20 updates and final 150; latest two complete native checkpoints |

No API judges the routes. Student inputs never contain hints or retrospective
labels. API failures, malformed hints, illegal suggestions or Teacher-context
overflow fall back to the ordinary static-skill Teacher; they do not wait forever
or disable GRPO. Hint coverage and fallback reasons are logged separately.

## Installation (Linux x86-64)

Two explicit environments keep the original SFT numerics and the native Ray/vLLM
runtime separate. Python 3.10 and 3.12, Git, a CUDA 12.8-compatible NVIDIA driver,
a CUDA toolkit for FlashAttention compilation, and Conda are prerequisites.
Do not run these commands on occupied GPUs. Both training stages target
**one node with 4×A100 80GB**: SFT uses four DDP ranks (2 samples/rank/update,
global8); stage2 uses FSDP4 and **TP2×DP2** rollout. Offline evaluation uses one
GPU. These are not single-24GB or multi-host memory-fit claims.

```bash
git clone https://github.com/leiye302/rethink-webshop-qwen25-3b
cd rethink-webshop-qwen25-3b
conda create -y -p "$PWD/.venv-opsd" -c conda-forge python=3.10 openjdk=11
conda activate "$PWD/.venv-opsd"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r configs/requirements-opsd.txt
python -m pip install --no-build-isolation flash-attn==2.8.3
mkdir -p assets
export PYTHONPATH="$PWD/src:$PWD/.deps/sdar"
python -m rethink_webshop.java --prepare
python -m rethink_webshop.setup --framework --data
python -m pip install -e .deps/sdar -c configs/requirements-opsd.txt
python -m pip check
export JAVA_HOME="$PWD/assets/java"
export JVM_PATH="$JAVA_HOME/lib/server/libjvm.so"
bash scripts/prepare.sh
python -m pytest -q
python scripts/smoke_native.py
```

`prepare.sh` fetches [public SDAR](https://github.com/ZJU-REAL/SDAR) at
`d511f043afe9199b8576fc99ea52e12d84841f32`, applies the 3-file patch, verifies
unchanged loss/skill/prompt sources, extracts the frozen SFT data, downloads
checksum-pinned WebShop assets and builds native Lucene indexes. All caches,
models and runs below this checkout are ignored by Git. No paid API or training
is started during preparation. See [the environment/resource details](docs/REPRODUCING.md).

## 3B: start from the existing SFT checkpoint

The original FP32 inference shards, tokenizer and configs are available as
[release assets](https://github.com/leiye302/rethink-webshop-qwen25-3b/releases/tag/route-sft1500-v1).
The downloader verifies **every file's size and SHA-256**, including the release
manifest. No optimizer/RNG states or training logs are published.
This is an initialization checkpoint, **not** an exact resume of SFT.

Read and accept the [Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/aa8e72537993ba99e69dfaafa59ed015b17504d1/LICENSE)
before downloading the 3B weights; they are **not Apache-2.0 model weights**.

```bash
python -m rethink_webshop.setup --checkpoint1500 --accept-model-license
export HINT_API_BASE="https://your-provider.example/v1"
export HINT_MODEL="your-text-model-id"
read -rsp 'Hint API key: ' HINT_API_KEY; echo
export HINT_API_KEY
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_opsd.sh --allow-api-cost
```

Select a compatible text model explicitly (e.g. a MiMo endpoint); no expensive
model is silently selected and no API credits are included. Use a process manager
you control for an unattended job. To resume the latest native checkpoint of the
same run, add `--resume`; never change the initialization/data/recipe in place.

## Optional: reproduce Route-SFT from base

```bash
python3.12 -m venv .venv-sft
.venv-sft/bin/pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.venv-sft/bin/pip install -r configs/requirements-sft.txt
python -m rethink_webshop.setup --base --accept-model-license
.venv-sft/bin/python -m rethink_webshop.audit_data
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_sft.sh
```

The script first runs disposable numerical/memory checks, reloads the pinned base,
then trains the unchanged loss/sampler. It saves complete resumable SFT checkpoints
at 750/1500 and evaluates validation loss, 200 empty-assistant-prefix generations
and the fixed 128 native episodes. Resume explicitly with
`bash scripts/train_sft.sh --resume runs/sft/checkpoint_000750` under the same
four-GPU allocation. SFT resume includes the four per-rank RNG states; old
single/two-rank optimizer checkpoints cannot resume this topology. The released
inference weights remain valid for fresh four-GPU stage2 initialization.
An optional newly trained initialization can be selected by
`--actor runs/sft/checkpoint_001500/model` for stage 2.

## Matched baseline and evaluation

```bash
# Same native GRPO+OPSD beta5, from base, with NO local hint API.
python -m rethink_webshop.setup --base --accept-model-license
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_opsd.sh --baseline --run-dir runs/baseline

# Frozen offline greedy evaluation, no API, same 128 goals and 200 source prompts.
CUDA_VISIBLE_DEVICES=0 bash scripts/eval.sh --model assets/base --output runs/eval_base
CUDA_VISIBLE_DEVICES=0 bash scripts/eval.sh --model assets/route-sft1500 --output runs/eval_sft1500
```

Offline greedy scores and online RL validation with native temperature 0.4 are
separate protocols, not interchangeable numbers. Success means native terminal
score=1; mean task score, protocol/reasoning rates and decision-level marker rate
are also reported. Marker rate is **not a fraction of all generated tokens** and
is never a training stop gate. Source Good/Bad generation groups are not labels
of the freshly generated reasoning, so they are not reported as route precision.

## Method and verification boundaries

- [Exact method/data/loss and information flow](docs/METHOD.md)
- [Minimal upstream differences](docs/UPSTREAM_DIFF.md)
- [Resource, dependency, evaluation and recovery details](docs/REPRODUCING.md)
- [Data notice](data/README.md) and [licenses](THIRD_PARTY.md)

CPU tests exercise the exact upstream Teacher builder, unchanged Student response
tokens, row identity after reordering, full-vocabulary BCE/masking, frozen data,
API failure fallback and source hashes. A clean-checkout CPU test is **not** a new
GPU/vLLM/150-update training reproduction. A real four-process CPU DDP test checks
the same SFT accumulation, updates and checkpoint/resume against serial execution;
see `docs/VERIFICATION.md` for the
checks actually executed. No 1.5B result is inferred from the released 3B weights.
