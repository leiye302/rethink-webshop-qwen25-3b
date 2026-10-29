# Resources, execution and recovery

## Public resources only

- SDAR commit and 3-file patch: `configs/upstream.json`.
- WebShop1000 catalog, attributes and human-instruction auxiliary file are
  downloaded from two pinned public dataset mirrors; all sizes/hashes and the
  official spaCy wheel are in `configs/assets.json`. Native synthetic goals
  remain selected (`human_goals=false`). Do not switch to a million products or
  a different goal generator and call it the same experiment.
- Base revisions/files: `configs/model.lock.json`, downloaded from public
  Hugging Face, verified against pinned LFS SHA256 or Git blob identity.
- Frozen data: `data/frozen_sft.zip`, byte hashes in `configs/data.json`.
- Exact3B SFT1500 inference export: public release; all21 assets are locked in
  `configs/checkpoint1500.json`. Original7 FP32 shards, no quantization/dtype
  conversion. No resharding is needed for inference initialization.
- External hint API is optional only for baseline/evaluation; method training
  needs an explicitly chosen compatible provider/model/key and cost consent.
  No private account, API balance or third-party service uptime is bundled.

New code is Apache2.0; model/data/SDAR upstream licenses remain separate.

## Environment and storage

Native stage2: Linux x86-64, Python3.10, Java11, CUDA12.8 torch2.8.0, vLLM0.11.0,
transformers4.57.3, Ray2.50; complete pinned Python lists in `configs`.
Install FlashAttention2.8.3 with the CUDA toolkit available. Check compiler/driver
compatibility before renting a long training session. SFT: Python3.12,
torch2.8.0+cu128, transformers4.57.1, BF16 autocast/FP32 master weights.
Do not silently switch optimizer precision to make a small GPU fit.

`python -m rethink_webshop.java --prepare` detects the documented Conda
OpenJDK11 layout (usually `.venv-opsd/lib/jvm`) and creates an explicit internal
`assets/java` link to the actual Java home after checking its JVM library/version.
It does not point outside the checkout or assume the prefix itself is Java home.
The native CPU worker reads checksum-verified public catalog files directly
inside `.deps/sdar`; no external data links or author-machine links are used.
All scripts resolve their own project root, so moving/clean-cloning is supported.

Approximate raw model/optimizer storage is not a guaranteed runtime-fit estimate:
3B published initialization is about12.34GB, base BF16 about6.17GB;
1.5B base about3.10GB. A full FP32 SFT checkpoint with AdamW states can require
about37GB for3B /19GB for1.5B, before temporary atomic-save space. Keep room for
both750/1500 and in-progress saves. RL checkpoint size depends on FSDP state
format; retention is latest2, not unlimited. Do not delete the only verified
resume state before a newer complete native checkpoint exists. Environment/CUDA
packages and caches require additional space; disk checks belong before launch.

Both training launchers now require4 explicitly assigned **A100 80GB** devices
on one node (at least75GiB visible VRAM each;40GB/MIG slices are rejected).
SFT uses DDP4, two micro-batch1 rows/rank/update and the same global8-row5:3 order.
Stage2 uses FSDP4 and TP2×DP2 rollout; PPO mini-batch64 and global Actor/rollout/
reference micro-batches16/32/32 are preserved by halving per-GPU micro-batches.
Native decision padding still uses32. Four GPUs do not increase16×8 episodes.
Baseline/method and both sizes share these hardware settings.

The128 train plus128 validation native environment actors reserve25.6 logical
Ray CPUs at0.1 each, before model workers/driver. Provide at least32 schedulable
CPUs and plan256–512GiB host RAM (prefer512GiB); actors load native product,
NLP and Java state. These are capacity recommendations, not measured minima.
SFT also replicates FP32 parameters/AdamW and can load4 CPU optimizer copies
during resume; host RAM matters as well as VRAM.

Four-A100 CUDA/NCCL/vLLM memory fit and throughput remain unverified without the
actual cards. The four-process CPU Gloo test checks the production partition,
loss scaling, updates and checkpoint/RNG resume, not BF16/NCCL bitwise parity.
Run the disposable four-GPU SFT preflight before training. Offline evaluation
still uses one GPU to preserve the fixed greedy protocol; other SFT ranks wait
at the boundary instead of writing duplicate evaluation evidence.

## SFT exact resume

The trainer first hashes base/data/code, checks both tokenizer boundaries and
the5:3 no-repeat order, checks numerical AdamW parity, then executes2 disposable
real-model distributed updates including a long balanced batch. Formal training reloads the
base afterward; those2 updates do not contaminate initialization. Preflight and
training must use identical code/config/data/parent files.

SFT750/1500 checkpoints contain model/tokenizer, optimizer, scheduler, four rank
RNG states, world-size binding, batch
order, immutable identity and SHA manifest. Pending directories are not resumes.
Only exact matching complete4-rank checkpoints may resume. `--resume` never restarts
from base silently. SFT inference-only release cannot resume an optimizer.

An optional shutdown guard uses `WEBSHOP_SAVE_BY_ISO` with explicit timezone;
the deadline is a wall-clock safety setting, not a scientific hyperparameter.
Without it, no automatic time limit is set. Evaluation evidence is atomic and
failed/pending runs are not reported as finished.

## Stage2 resume and hint safety

Native SDAR owns FSDP saving/resume, rewards, all GRPO updates and native gate.
Run `train_opsd.sh --resume --allow-api-cost` against the same directory only.
The wrapper rejects a different Actor/config/validation/world-size identity and
requires all4 native model/optimizer/extra-state rank shards plus dataloader state.
Old single/two-rank optimizer states cannot resume this topology; inference-only
model exports remain valid fresh initialization weights. Its default
initialization is not “latest weights from another run.” Native latest2 retention
keeps final150 as one of the latest2. No cross-server backup agent is required.

Keys are entered via environment, then written to an ignored owner-only600
file under that run and removed from the launch environment before Ray startup.
Only the **path** to this file is passed through Ray configuration, never the key.
It is deleted on normal shutdown; after an abrupt host/process crash, an owner
may remove the remaining private file. Never publish `runs/`, `.env` or API logs.
Transport uses bounded HTTP/1.1 requests and controlled error codes; HTTP bodies
and secret-bearing exception strings are not printed.

One failed hint must not stop training: unavailable credit/auth, transport, empty
content, missing/duplicate step IDs, illegal next commands, or prompt overflow
produce ordinary static-skill Teacher fallback and coverage counters. A high
fallback rate means the **local-hint intervention is poorly applied**, not a
successful hint-quality result; report coverage and investigate separately.
Syntactic legality is not proof of semantically helpful hints.

## Native evaluation and metrics

The same fixed128 goal IDs are selected by
`numpy.random.RandomState(0).choice(range(500),128,replace=False)`, all with
native synthetic goal seed0. SFT source tasks live at500+. The raw public code
has randomized validation and worker-dependent goal seeds; this matched protocol
fix is applied equally to both model sizes and baseline/method.

Native RL validation uses sampling/temperature.4, max15 actions, response512;
offline SFT evaluation is greedy. The Actor always starts an empty assistant
prefix for free generation; no stored source R, target marker, correction or
future feedback is prefilled. The fixed200 prompt subset does include source
Good/Bad provenance but does not classify new R automatically.

Native success is `done and raw_score == 1.0`, sparse training reward10 for
success/0 otherwise. Score summaries can be expressed as raw0..1 or percentage;
label the scale. Report both score and success, as partial matches need not be
successes. RETHINK rate counts decisions with at least one generated marker ID
divided by active decisions, independent of text-special-token filtering.
The native projector is unchanged; stricter protocol metrics are telemetry,
not a new action acceptance/retry policy.

The optional offline evaluator preserves the previously used guard against
environment-internal `<q>/<a>/<c>/<r>` search shortcuts; ordinary visible
navigation buttons such as `click[< Prev]` remain allowed. A generated hidden
operator is recorded as an evaluation error, never silently scored as a normal
command. Native stage2 scoring/parser itself is not modified by this guard.
