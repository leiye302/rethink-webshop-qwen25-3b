# Active 3B Stage1 data: buy-v9

The 3B repository now runs **public Qwen2.5-3B base → new Route-SFT1500 →
GRPO+OPSD150**. Its active `data/frozen_sft.zip` and `configs/data.json` are
the exact buy-v9 database already published by the 1.5B companion at commit
`b7332bfe04ae68d5462b8e921c2ae8816b23744b`, under `data/buy_v9/`.
Both files are bundled here; that other checkout is never a runtime dependency.

| Partition | Good | Bad |
|---|---:|---:|
| Train | 7,500 | 4,500 |
| SFT validation | 631 | 271 |
| Free-generation diagnostic | 100 | 100 |

There are268 Buy targets:176 Good,92 Bad. The accepted revision adds87 natural
Good Buy and30 corrected Bad Buy rows relative to its preceding library,
replacing equal numbers of same-route non-Buy examples. Good still keeps the
original sampled action; Bad still targets marker plus accepted correction.
Validation and the200-H diagnostic are unchanged. No API is needed to obtain
the frozen data. The original publication checks are in
`data/buy_v9/verification.json`; those are data checks, not a trained-model score.

- ZIP SHA256: `34b25a42c2717bdee5d3a3bee8a955b4cffcdcc60ebbbfa3ca62a9c6693692f9`.
- Train SHA256: `860a7b0ac73f91396089365947570c2f3052b2318d553ec1ac204964e78b9e59`.

## Start a fresh data-bound experiment

Use a fresh clone of the current3B repository and follow the main README.
No manual ZIP/config copying is needed: buy-v9 is already active. Preparation
verifies the ZIP and every member before using them. It refuses to replace a
different previously extracted `datasets/frozen_sft` directory.
The actual Stage1 validator also compares the extracted manifest against the
active data lock, so bypassing preparation cannot silently train old cached rows.

If an old checkout has data/runs, keep it intact and clone a new directory for
the new experiment. Do not delete old records, overwrite extracted old data,
resume old optimizer states or relabel the old SFT release as buy-v9.

After preparation, model-license acceptance and installation:

```bash
.venv-sft/bin/python -m rethink_webshop.audit_data
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_sft.sh
# Configure your Teacher-only hint API as documented, then:
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_opsd.sh --allow-api-cost
```

Stage2 defaults to `runs/sft/checkpoint_001500/model` and verifies its Stage1
completion/parent/recipe/data/model identity. The old inference-only
`route-sft1500-v1` release is retained for historical use, not substituted for
this new training. Baseline still starts from the public3B base, and independent
evaluation does not require an SFT provenance check.

## Recipe unchanged apart from the selected data and initialization workflow

SFT remains1500 updates,5 Good/3 Bad,global8,micro1,LR2e-6,masked H/R,
suffix CE plus0.1 independent full-vocabulary marker BCE,no reference KL,
max_length8192/error,and four DDP ranks. Stage2 remains150 updates,Actor-owned
RETHINK routing,current-Actor no-gradient Teacher,Teacher-only local hints,
SDAR beta5 and coefficient0.01. The fixed140 task IDs,seeds,decoding and scores
are unchanged. No new trigger-rate or benchmark-performance gate is introduced.
