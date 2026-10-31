# ALFWorld-final method alignment

This release uses the final ALFWorld Actor-routed local-hint method, not its
annotation model or an earlier always-fallback API draft. Both WebShop model
sizes use identical hint/loss code. No formal GPU result follows from this document.

## Fixed shared algorithm

- Every trajectory participates in native SDAR GRPO + OPSD.
- Only the Actor's valid generated RETHINK token IDs select hint targets.
- The API neither labels Good/Bad nor approves/rejects Actor routes.
- Static public WebShop skills apply to every Teacher decision; triggered
  decisions additionally receive the accepted local reasoning skill.
- Teacher is a no-gradient view of the SAME CURRENT pre-update Actor, scoring
  the SAME sampled Student response. It is not the API model or a frozen initial
  Teacher. Student/reference/evaluation never receive local hints.
- Gate beta=5, distillation coefficient=.01, native GRPO, reference KL=.01 and
  native entropy/masking/clipping remain unchanged. SFT remains suffix CE +
  .1 full-vocabulary marker BCE, without SFT KL. Existing data and SFT recipe
  are not rebuilt by this alignment.

## Reused code, not a separately invented prompt/parser

`rethink_api_protocol.py` and `rethink_benchmark_contract.py` are text-identical
to the final ALFWorld release, including its existing WebShop adapter.
`configs/hint_protocol.json` pins their SHA-256 values. The shared protocol fixes:

- model `glm-5.3-flash`, temperature0, `reasoning_effort=low`;
- `max_tokens=min(16384,1024+192*number_of_targets)`, JSON output;
- one complete chronological trajectory per request, all its Actor-selected
  targets merged, exact raw pre-action target snapshots repeated;
- compact repeated observations and deduplicated command sets;
- output `{"hints":{"EXACT_STEP_ID":"Next command: 'NATIVE_COMMAND'. Brief guidance."}}`;
- exact ID coverage, duplicate-key rejection, native legal command checking,
  identical hint length and boundary checks. Only the pinned GLM generation path.

A completed but malformed hint falls back to the ordinary static-skill Teacher.
Transport/auth/quota failure instead retains the SAME rollout/pre-update Actor
and waits to retry; it does not silently train without the intervention.
Six bounded HTTP attempts precede a retained-batch wait (60s transient,
300s auth/quota/request hold), as in ALFWorld final. Successful calls are cached.
A hint cut by the native Teacher context budget is a visible error, not a
silently discarded or falsely counted hint.

## Necessary WebShop adaptation

Only environment-specific evidence/actions are adapted. Native raw current
pages, available click/search commands, task and executed action are recorded.
WebShop's public projector mutates its input response list in-place, so reasoning
must be decoded from actual valid generated IDs, NOT read back from that list
after execution. This avoids an empty/incorrect reasoning payload.
Reasoning text is stripped exactly as in ALFWorld final. If the model omits a
complete think block, the recorded response is retained as the diagnostic
reasoning fallback, rather than replacing its evidence with an empty string.

WebShop retains its own native 1000-product/synthetic-goal environment,
reward, history, step budget, static shopping skills, fixed validation goals
and registered model-size initialization. It does not receive ALFWorld kitchen
commands, its task count or ALFWorld's evaluation set. Public upstream loss,
Actor update, reward/projection, native prompt and memory files remain locked.

## Runtime engineering corrections

The shared environment originally pinned datasets4.4.1 together with
pyarrow19.0.1, which cannot resolve: that datasets release requires pyarrow>=21.
The install now pins pyarrow21.0.0 without changing data or loss.

Native WebShop spaCy3.7.2 needs typer<0.10, while FastAPI's optional command-line
extra needs newer typer. `scripts/install_opsd.py` keeps the native scorer/NLP
versions fixed, installs vLLM0.11 and all actual FastAPI server dependencies,
and omits the unused FastAPI CLI. The text-only task also does not request
the unrelated optional Decord video backend (whose published wheel has an
incompatible CPython3.6 tag). It does not patch package metadata or
library source. Run pip check and the real GPU/native-environment smoke before
claiming a deployment works.

API requests retain TLS certificate verification and reject redirects.
OpenCode Go receives its required non-secret session header and an explicit
client User-Agent. Credentials stay in an owner-only file and are never Hydra
values or process arguments. The launcher rejects a substituted hint model and
binds resume identity to the hint source hashes. Missing indexed model shards
are rejected before training.

## Verification boundary

CPU tests and a real GLM API request do not establish four-GPU FSDP/NCCL,
TP2xDP2 vLLM rollout or a complete 150-update run. A one-GPU numerical fixture
must be reported separately from those tests and from benchmark performance.
