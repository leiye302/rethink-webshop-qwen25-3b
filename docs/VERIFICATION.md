# Publication verification

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
- The subsequently added native-resume-file guard was tested locally. The full
  current test suite has22 tests; clean-checkout GitHub CI runs it with CPU torch.
- Tests cover full-vocabulary BCE/target shift/masking, no-repeat5:3 sampler,
  data identities/task split, Good original actions, valid marker-ID capture,
  row permutation/duplication, current-Actor scoring interface, unchanged Student
  responses/inputs, per-row Teacher budget fallback, partial hint validation,
  bounded timeout/auth fallback and missing native-resume files.
- All21 released3B assets were uploaded, byte-count/SHA256 matched against the
  exact original1500-step export and independently matched to GitHub's asset
  digests before the release was made public.

`scripts/check_release.py` and `.github/workflows/ci.yml` make these checks
repeatable. Test doubles validate information flow; they are not real distributed
Actor/vLLM training. No fixed initial Teacher is introduced by those doubles.

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
