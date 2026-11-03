# Minimal differences from SDAR

Pinned source: `ZJU-REAL/SDAR@d511f043afe9199b8576fc99ea52e12d84841f32`.
The patch is only 3 files; the complete diff is `patches/sdar.patch`.

| Source | Named change |
|---|---|
| `rollout_loop.py` | Retain special tokens; capture Actor marker IDs, pre-action state/commands and feedback; attach episode/step identity before native flattening |
| `skillsd_ray_trainer.py` | Select ordinary native Teacher scoring or extra-context scoring; native losses/advantages stay unchanged |
| `main_sdar.py` | Import a read-only validation-telemetry subclass; inherited training/Teacher/loss unchanged |
| `webshop/envs.py` | Filter content-disjoint goal pools and use the shared140 validation tasks/seeds; native training sampling is retained within the allowed pool |

Extra code lives in `src/rethink_webshop`, not copied modifications throughout
the optimizer. `sdar_utils.py`, `dp_actor.py`, `core_algos.py`, native Teacher
batch builder, static skills, WebShop prompt/memory/parser and score/reward are
byte-hash locked and unchanged. The source verifier rejects additional tracked
changes and checks reverse application of the patch.

The API resilience update touches only that wrapper's transport/batch waiting,
explicit proxy forwarding and diagnostics. Permanent configuration/local-I/O
errors fail visibly; transient outages retain the same batch. The public SDAR
patch and source locks, API model/body/parser, all SFT/RL loss and recipe values,
and shared140 manifests are unchanged. See `API_TROUBLESHOOTING.md`.

Configuration changes versus `examples/grpo_opsd_trainer/run_webshop_3b.sh`:

- beta0 → **5**, for both baseline/method (OPSD coefficient remains .01).
- Actor initialization: newly trained local Route-SFT1500 before Stage2, for3B
  as for1.5B. The3B default uses the published buy-v9 profile; select the same
  profile in1.5B for matched data. Baseline still starts the corresponding base.
- Actor-marker Teacher-local hints enabled only for method.
- Explicit150 updates, validation every10, save every20/final150, latest2.
- Shared140 content identities and worker seeds for all method/size comparisons.
- Console logging/local output paths; no account tracking key is required.
- Single-node4×A100-80GB profile: native FSDP4, rollout TP2×DP2; Actor micro4/GPU
  and rollout/reference logprob micro8/GPU retain the old global micro-batches
  and32-decision padding divisor. The public losses remain byte-identical.
- Native placeholder text rows are produced directly instead of downloading
  unrelated Geometry3K images. Their fields/counts are the public text recipe;
  actual shopping tasks continue to come from native environment resets.

No generic benchmark, extra episode Analyzer, Oracle online router, frozen
initial Teacher, forced marker, new countdown prompt, dynamic trigger gate,
different reward, auxiliary marker BCE during RL or hint-visible Student is added.

SFT reference loss/sampler/token code is copied byte-for-byte from the frozen
training implementation, verified by `configs/reference.lock.json`. Portable
training/evaluation wrappers change deployment paths, selected base revision and
DDP transport, not loss, quotas or schedule. The named subsequent3B data revision
is buy-v9, not an unreported reannotation or recipe change; see DATA_BUY_V9.md.
Its Stage1 validator rejects stale extracted data, and its Stage2 handoff checks
the completed local SFT's parent/recipe/data/model identity. No loss is added.
Four SFT ranks each accumulate
2 rows from the same original8-row global batch. Scaling4/8 followed by DDP rank
averaging preserves the sample mean. Checkpoints bind all four rank RNG states
and world size; rank0 alone writes/evaluates at each boundary.
The shared scientific losses and batches remain identical across sizes; the
base locks differ, and dataset selection must explicitly match the buy-v9
profile when comparing the current3B default to1.5B.
