# Publication verification

The historical CPU/CI checks below refer to their stated pre-alignment release.
For the final-ALFWorld GLM protocol alignment, install/runtime corrections and
the boundary of new single-GPU fixtures, see `ALFWORLD_ALIGNMENT.md`.
Do not treat the historical CI links as a new four-GPU execution of changed code.

## Final-ALFWorld alignment checks

- Both model-size source trees passed all **39 Linux CPU tests**, including
  four-process Gloo SFT update/checkpoint-resume parity, the exact upstream
  Teacher builder, generated-token route capture, and final GLM request/parser
  and transport-hold contracts. Public CI reruns these checks on each commit.
- The two hint protocol sources are byte-identical to the final ALFWorld
  implementation and are checked against `configs/hint_protocol.json`.
  SFT loss/sampling, native GRPO/OPSD loss and the Teacher builder are unchanged.
- All 12,000 training rows, 902 validation rows and 200 generation-diagnostic
  rows passed the released 3B tokenizer boundary scan, with zero boundary
  mismatches or encoding errors and no training/validation task overlap.
  This is a 3B tokenizer check, not a 1.5B forward or training result.
- Separate 1.5B single-GPU diagnostic fixtures exercised a full-parameter
  SFT update/save/reload, native GRPO+OPSD backward with recorded responses and
  fixture rewards, a real GLM Teacher-hint scoring request, and vLLM generation.
  Native WebShop reset/search/feedback and runtime `pip check` also passed.
  These are diagnostic checks, not an on-policy training run or benchmark score.
- Complete 3B GPU forward/backward and four-A100 FSDP/TP2xDP2 execution remain
  unverified. Do not substitute the 1.5B diagnostics for those missing checks.

## Executed without GPU or paid hint requests

- Source/data allowlist, syntax and current/history privacy scans passed.
  Archive contents are inspected, not merely ignored by `.gitignore`.
- Both repositories' SFT data ZIPs and training row hashes match; only the
  base revision differs in their scientific SFT configurations.
- Exact public SDAR commit and forward/reverse application of the3-file patch
  verified. GRPO/OPSD loss, native prompt/parser/score and skill hashes match
  unchanged upstream blobs (not platform-dependent line-ending guesses).
- Linux CPU execution on independently exported source trees: **21 tests passed
  for each model-size repository**, including the exact unmodified upstream
  Teacher batch builder executed with a CPU-only RPC double.
- The original22-test suite, including the native-resume-file guard, passed
  in **clean-checkout GitHub-hosted Ubuntu CPU CI for both repositories**.
  CI installs CPU torch, fetches the public SDAR pin from GitHub, extracts the
  bundled data, checks source locks, executes the tests and scans Git history.
- Tests cover full-vocabulary BCE/target shift/masking, no-repeat5:3 sampler,
  data identities/task split, Good original actions, valid marker-ID capture,
  row permutation/duplication, current-Actor scoring interface, unchanged Student
  responses/inputs, Teacher budget checking, hint validation,
  timeout/auth handling and missing native-resume files. The final alignment
  now uses explicit API holds and rejects silently truncated Teacher hints.
- All21 released3B assets were uploaded, byte-count/SHA256 matched against the
  exact original1500-step export and independently matched to GitHub's asset
  digests before the release was made public.

`scripts/check_release.py` and `.github/workflows/ci.yml` make these checks
repeatable. Test doubles validate information flow; they are not real distributed
Actor/vLLM training. No fixed initial Teacher is introduced by those doubles.

## Four-A100 profile

The4×A100-80GB update adds guards for device count/model/visible VRAM, native
global-microbatch/padding equivalence, rejection of old two-rank resumes, and
a real4-process CPU Gloo test using the production SFT update/checkpoint code.
That test compares three sample-mean CE+BCE/AdamW updates against serial8-row
execution, with a complete four-rank RNG/optimizer/scheduler checkpoint/resume
between updates2 and3. **All28 tests passed with no skips in clean Ubuntu CPU CI
for both variants**, including the actual four-process Gloo test:
[3B](https://github.com/leiye302/rethink-webshop-qwen25-3b/actions/runs/40848789893),
[1.5B](https://github.com/leiye302/rethink-webshop-qwen25-1.5b/actions/runs/40848785475).
This is not a CUDA/NCCL/A100 memory or throughput test. The raw3B SFT1500 weights
and frozen datasets are unchanged by the hardware adaptation.

## Not established by these publication checks

- A fresh1.5B SFT run, either model's fresh150-update RL run, CUDA numerical or
  memory fit, distributed Ray/vLLM startup, or task performance from this new
  checkout. Training preflight and the documented GPU runs are still required.
- API semantic quality or production coverage with an operator's chosen provider.
  Tests use simulated responses and failures, not paid quality calls.
- A new full native WebShop installation/reset/index build on the publication
  test machine. The supplied `scripts/smoke_native.py` exercises real native
  reset→action→feedback after installing/downloading the documented public assets.

The exact3B initialization is a real previously trained checkpoint; shipping it
does not make the new stage2 recipe a completed training result. No1.5B score
is advertised. Check the public CI run for the current commit before a long GPU
job, then run the real native CPU smoke and required GPU preflight.
