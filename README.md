# RETHINK WebShop — Qwen2.5-3B

> Shared cross-method evaluation: [exact common140 tasks and unchanged sampling](docs/SHARED_EVALUATION140.md).


An independently runnable **3B base → Route-SFT → GRPO+OPSD** recipe.
Route-SFT is followed by **public SDAR GRPO+OPSD (gate beta 5)**, with extra
Teacher-only local hints at decisions where the **Actor itself generates
`<RETHINK>`**. Run Stage1 on the current **buy-v9** database, then initialize
Stage2 from that run's complete **1500-step local checkpoint**, not an old release,
SEED/M1 or an Analyzer checkpoint.

The [1.5B companion](https://github.com/leiye302/rethink-webshop-qwen25-1.5b)
also starts from its public base and trains SFT first. The exact same buy-v9
profile is published there under `data/buy_v9`; select that profile explicitly
when matching data across sizes. Here buy-v9 is already the active default.
Neither repository needs the authors' computer, servers, private datasets or
preinstalled environments. Model weights are separate downloadable assets.
See [the versioned data and fresh-run instructions](docs/DATA_BUY_V9.md).

## What is fixed

| Component | Recipe |
|---|---|
| Base | Qwen2.5-3B-Instruct, pinned revision in `configs/model.json` |
| SFT data | buy-v9: 7,500 natural Good + 4,500 reviewed Bad; matches the 1.5B buy-v9 profile |
| SFT | 1 epoch / 1,500 updates; every batch 5 Good + 3 Bad; full parameters |
| SFT loss | Sample-mean route/action/EOS CE + 0.1 independent full-vocabulary marker BCE; **no reference KL** |
| Stage 2 | 150 GRPO+OPSD updates; beta=5, distillation coefficient=0.01 |
| Teacher | Current Actor weights, no-gradient scoring view; **not a frozen initial Teacher** |
| Hints | Static public SDAR WebShop skills everywhere; additional local API hint only at Actor-marker decisions |
| Environment | SDAR default 1,000 products + synthetic goals, history 2, at most 15 actions |
| Budgets | Prompt 4,096 / response 512; unchanged prompt/action parser/reward |
| RL validation | Same shared fixed140 content goals for both arms/sizes, every 10 updates |
| RL checkpoints | Every 20 updates and final 150; latest two complete native checkpoints |

No API judges the routes. Student inputs never contain hints or retrospective
labels. The final ALFWorld hint model and exact protocol are pinned to
`glm-5.3-flash`; malformed/illegal hints fall back to static-skill Teacher.
Transient transport/quota failures retain the same batch and wait, rather than
silently dropping hints. Permanent configuration/auth and local disk errors stop
visibly without an Actor update or repeat billing. Teacher-hint truncation fails visibly. See
[the exact alignment and runtime corrections](docs/ALFWORLD_ALIGNMENT.md).

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
python -m pip install --upgrade pip setuptools wheel packaging==25.0
python scripts/install_opsd.py
python -m pip install --no-build-isolation flash-attn==2.8.3
mkdir -p assets
export PYTHONPATH="$PWD/src:$PWD/.deps/sdar"
python -m rethink_webshop.java --prepare
python -m rethink_webshop.setup --framework --data
python -m pip install --no-deps -e .deps/sdar
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

## Stage1: retrain Route-SFT from the 3B base

Read and accept the [Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/aa8e72537993ba99e69dfaafa59ed015b17504d1/LICENSE)
before downloading the 3B weights; they are **not Apache-2.0 model weights**.

```bash
python3.12 -m venv .venv-sft
.venv-sft/bin/pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.venv-sft/bin/pip install -r configs/requirements-sft.txt
python -m rethink_webshop.setup --base --accept-model-license
.venv-sft/bin/python -m rethink_webshop.audit_data
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_sft.sh
```

The unchanged SFT script first runs disposable numerical/memory checks, reloads
the pinned base, then trains 1500 updates. It saves complete resumable SFT
checkpoints at750/1500 and evaluates validation loss,200 truly free generations
and the shared140 native episodes. Resume only this same data/recipe/world size
with `bash scripts/train_sft.sh --resume runs/sft/checkpoint_000750`.
Do not resume an optimizer trained on the old database, overwrite old extracted
data or relabel an old model as buy-v9. See [fresh-run migration](docs/DATA_BUY_V9.md).

## Stage2: initialize from the newly trained SFT1500

```bash
export HINT_API_BASE="https://your-provider.example/v1"
export HINT_MODEL="glm-5.3-flash"
read -rsp 'Hint API key: ' HINT_API_KEY; echo
export HINT_API_KEY
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_opsd.sh --allow-api-cost
```

The default Actor is **`runs/sft/checkpoint_001500/model`**. Stage2 verifies the
complete Stage1 manifest,1500 updates,3B parent,current SFT recipe,buy-v9 row
hashes and model file hashes. Missing/intermediate/old-data checkpoints are
rejected; there is no automatic fallback to the old released weights.
Only the model initializes RL: the SFT optimizer is not restored into GRPO.

After preparation and API configuration, the two stages can be run sequentially:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_sft.sh &&
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_opsd.sh --allow-api-cost
```

Stage2 starts only after Stage1 exits successfully. No trigger-rate or
performance threshold is added. Model size,losses,batches,RL gate and shared140
evaluation are unchanged by this pipeline/data-selection update.

Use a compatible provider serving the fixed `glm-5.3-flash` model and final
ALFWorld request parameters; do not substitute the annotation model.
No key or API credits are included. Use a process manager
you control for an unattended job. To resume the latest native checkpoint of the
same run, add `--resume`; never change the initialization/data/recipe in place.

Before allocating GPUs, check the configured API and writable cache with no model
or SDAR installation required:

```bash
PYTHONPATH="$PWD/src" python -m rethink_webshop.hint_api --check
# Explicit opt-in: one fresh synthetic request through the training transport/parser.
PYTHONPATH="$PWD/src" python -m rethink_webshop.hint_api --check --allow-api-cost
```

The first command makes no network request. Neither command trains or uses CUDA.
Do not put keys in command arguments. Proxy use is unchanged by default; opt out
only if appropriate for your network with `RETHINK_HINT_TRUST_ENV=false`.
See [API diagnosis, failure handling and update/resume boundaries](docs/API_TROUBLESHOOTING.md).

## Historical released weights

The original [route-sft1500-v1 release](https://github.com/leiye302/rethink-webshop-qwen25-3b/releases/tag/route-sft1500-v1)
retains its original FP32 inference weights,tokenizer and SHA-locked provenance.
Those weights used the prior database; they are not the result of this new
buy-v9 Stage1 and are not accepted as its Stage2 start. The historical resource
and license information remain available; no old model is relabeled or deleted.

## Matched baseline and evaluation

```bash
# Same native GRPO+OPSD beta5, from base, with NO local hint API.
python -m rethink_webshop.setup --base --accept-model-license
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_opsd.sh --baseline --run-dir runs/baseline

# Formal native shared140 evaluation, no API, shared140 goals; native vLLM, temperature0.4.
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval.sh --model assets/base --output runs/eval_base
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval.sh --model runs/sft/checkpoint_001500/model --output runs/eval_sft1500
```

Formal evaluation and RL validation use the same shared140 native protocol:
temperature0.4, sampling, top-p1, unrestricted top-k, one rollout/task,
history2, prompt4096, response512 and15 environment steps. No hint API is used
at evaluation. The200-H greedy generation probe is a separate SFT diagnostic.
Old128/greedy scores are not relabeled. See the exact shared contract above.
Report native success and mean raw task score separately. Marker rate counts
marker-bearing active decisions, not a fraction of generated tokens. Source
Good/Bad groups are not labels for freshly generated reasoning.

## Method and verification boundaries

- [Exact method/data/loss and information flow](docs/METHOD.md)
- [Minimal upstream differences](docs/UPSTREAM_DIFF.md)
- [Resource, dependency, evaluation and recovery details](docs/REPRODUCING.md)
- [Data notice](data/README.md) and [licenses](THIRD_PARTY.md)

CPU tests exercise the exact upstream Teacher builder, unchanged Student response
tokens, row identity after reordering, full-vocabulary BCE/masking, frozen data,
API hold/fallback boundaries and source hashes. A clean-checkout CPU test is **not** a new
GPU/vLLM/150-update training reproduction. A real four-process CPU DDP test checks
the same SFT accumulation, updates and checkpoint/resume against serial execution;
see `docs/VERIFICATION.md` for the
checks actually executed. No 1.5B result is inferred from the released 3B weights.
