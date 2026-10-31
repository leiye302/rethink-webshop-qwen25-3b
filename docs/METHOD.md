# Method and data contract

## Stage 1: Route-SFT

The policy starts directly from the pinned public Qwen2.5 Instruct model. No
SEED Stage-1, M1, episode Analyzer or decision Analyzer SFT is a hidden prerequisite.
The frozen dataset in this release was authored from ordinary **3B** online
trajectories without RETHINK augmentation. Both model sizes deliberately reuse
those same samples; this is not claimed to be 1.5B on-policy collection.

The Actor sees the native task/history/current observation, followed by its
recorded natural reasoning. No source action, future observation, hidden goal,
review explanation or correction metadata is inserted into the conditioning
input. A sample uses:

```text
prompt:            original native visible H
assistant_prefix:  <think>original R</think> + its actual natural separator
Good response:     <action>original sampled A</action>
Bad response:      <RETHINK><action>verified corrective A*</action>
```

Good always retains the original natural action. A Bad corrective action may
equal the original action if that action is independently acceptable; changing
the action is not the meaning of a reasoning-error label. Unknown labels and
unsupported zero-separator merged-token boundaries do not enter the frozen
training set. Corrections are next-step WebShop commands, not whole plans.

The intended Bad semantic is a material reasoning error supported by currently
visible evidence, still uncorrected in the reasoning, that can mislead the next
decision (state/entity/goal confusion or violated action prerequisites). Failure
of the whole episode, lack of immediate progress, unsuccessful exploration, or
action quality alone is not sufficient. H includes only the actual native
history window; facts outside the Actor input cannot justify a detectable error.
Labels are data-authoring annotations, not an online Analyzer gate. The supplied
accepted library is frozen, rather than regenerated differently by each model.

Purchase-target cleanup was applied only to corrective actions. Targets can buy
when visible requirements/options support purchase, without privileged “steps
remaining” messages. No benchmark prompt or extra countdown was added. Good
actions were not silently replaced by Oracle purchases. Exact current-state
admissibility/replay evidence is stored separately from model conditioning.

### Dataset and order

| Partition | Good | Bad | Use |
|---|---:|---:|---|
| Train | 7500 | 4500 | One epoch, no cycling or dropping |
| SFT validation | 631 | 271 | Loss monitoring only |
| Fixed free-generation subset | 100 | 100 | Subset of SFT validation; empty assistant prefix |

Train/validation task identities and task hashes are disjoint. The native
benchmark pool 0..499 is separate from collected training tasks (500+). Complete
source row fields are included unchanged in the ZIP: hashes are in
`configs/data.json`. Train SHA-256 is
`db060c4703fa0b0fba767696d32597be6d2bc22d7e9067ca054c6b0776180888`.
The ZIP contains no server paths, API credentials, private logs or optimizer states.

The original task/route-balanced sampler (`src/sft_reference`) visits every row
once, uses seed=0, and forms 1500 batches of 5 Good/3 Bad. Four DDP ranks take
two micro-batch-1 samples each from the SAME original global batch; no per-rank
Good/Bad quota, cycling or new shuffle is introduced. Each local loss is scaled
by4/8 before DDP's rank-average, giving the same eight-sample mean gradient:

$$\frac{1}{4}\sum_{r=0}^{3}\sum_{i\in B_r}\frac{4}{8}\nabla L_i
=\frac{1}{8}\sum_{i\in B}\nabla L_i.$$

Full trainable FP32 parameters, BF16 autocast,
non-reentrant checkpointing, native AdamW with LR 2e-6, betas (0.9,0.95), weight
decay .01, epsilon 1e-8, gradient clip 1, warmup150 and cosine schedule1500 are
preserved. Maximum sequence length8192; overlength is an error, never truncation.

### Exact loss and boundary

Let s be H plus masked R and its original separator, u the route/action/EOS
suffix, and y=1 for Bad. The first supervised target is directly `<RETHINK>`
or the first token of `<action>`; the predictive logits are at suffix_start−1.

$$L_{suffix}=-\frac{1}{|u|}\sum_{t=1}^{|u|}\log\pi_\theta(u_t\mid s,u_{<t}).$$

$$r=z_{RETHINK}-\operatorname{logsumexp}_{v\ne RETHINK}z_v,\quad
L_{route}=\operatorname{BCEWithLogits}(r,y),\quad L=L_{suffix}+0.1L_{route}.$$

These are averaged over samples, not all concatenated reasoning tokens. No
reasoning CE or reference KL is used. The BCE repeats/reweights part of the CE
intentionally; it is a full-vocabulary marker-vs-rest probability, not sigmoid
of the raw marker logit or a two-token renormalization. H/R token targets have
zero CE, but shared model parameters still receive suffix gradients: this is
**not** a mathematical guarantee of unchanged free-generation reasoning.
The full H/R/suffix joint tokenization must equal the concatenated encoding;
the merged `><` path must not be silently accepted. EOS remains supervised.

`<RETHINK>` is registered once in available preallocated vocabulary space;
unexpected embedding resize is rejected. Decoding retains special tokens and
telemetry counts the generated marker ID rather than relying on filtered text.

## Stage 2: minimally adapted public GRPO+OPSD

All trajectories participate in native GRPO. Static public SDAR WebShop skills
are used for Teacher context on every step. A local hint is added only when the
same Actor's generated valid response IDs contain the registered marker.
Neither the hint model nor an Analyzer decides Good/Bad or whether to route.

After rollout, one API request groups the triggered steps of an episode. The
full episode, target pre-action visible states, available actions and later
feedback are privileged hint-authoring context. Each hint addresses the local
reasoning/state error and suggests one legal next command. A suggestion is not
executed and never replaces a sampled action, reward, trajectory or training
target. Only current legal clicks/open-query search grammar are accepted.

The Teacher is a **no-gradient forward view of current Actor weights**. For the
Student's exact sampled response prefix and next token, Teacher scores the same
token with static skill (+ applicable local hint); it does not generate a new
trajectory. The native confidence gate is unchanged:

$$\Delta_t=\log\pi_\theta(y_t\mid H,skill,hint,y_{<t})
-\operatorname{stopgrad}\log\pi_\theta(y_t\mid H,y_{<t}),\quad
g_t=\operatorname{stopgrad}\sigma(5\Delta_t).$$

Teacher log probabilities are detached. The public SDAR auxiliary term is the
native masked aggregate of $g_t(\log p_T-\log p_S)$ with coefficient .01, added
to the original GRPO/PPO objective. The public response mask and aggregation are
unchanged: this release does not introduce an action-only RL objective.
The native separate reference-policy KL (.01, low_var_kl) remains enabled in
stage 2; it is distinct from the absent SFT reference KL.

Batch16 tasks × group8 =128 rollout episodes/update; native max15 steps,
history2, prompt4096/response512, Actor LR1e-6, PPO minibatch64/micro4 per GPU,
native invalid-action penalty .1, native entropy coefficient and GRPO advantage
normalization unchanged. The public GRPO+OPSD launcher uses beta0; **beta5 is
the explicit requested change**, shared with the provided baseline.

The four-A100 profile uses native FSDP world4 and rollout TP2×DP2. PPO mini-batch
stays64; per-GPU Actor micro-batch8→4, rollout/reference logprob micro-batches
16→8 preserve their global sizes16/32/32. Native `adjust_batch` still rounds to
multiples of32 decisions, rather than adding more duplicate rows at world4.
Both comparison arms use the same changes. Different collective reduction order
and rollout worker assignment can change rounding/sampled trajectories; this is
not a claim of bitwise-identical two-versus-four-GPU runs.

Local hints never reach Student generation, Student forward, reference forward
or evaluation. Metadata is attached before flattening, then follows native
row balancing/duplication. The exact ALFWorld-final protocol fixes
`glm-5.3-flash`, temperature0, reasoning_effort=low and
max_tokens=min(16384,1024+192*target_count). Six bounded transport attempts
precede a retained-batch wait; auth/quota failure does not silently drop hints.
Requests are cached on exact protocol/model/body/endpoint identity and grouped
per complete episode. Output is an exact step-ID-to-string object, not an array.
Schema/ID failure falls back for the requested group; a bad command/length only
rejects that hint. A hint truncated by the native Teacher budget fails visibly.
Neither case changes the Actor route, executed rollout, reward or generation
budget. See `ALFWORLD_ALIGNMENT.md` and the pinned protocol source hashes.

## Evaluation interpretation

All arms/sizes use the fixed 128 reserved synthetic goals in `configs/eval128.json`,
seed0 for goal identity, native score and success definition. Fixed sampling
and matched validation seed are explicit protocol changes versus repeatedly
sampling the public validation pool. They apply equally to baseline/method.

Online RL validation: before training and every10 updates/final150, native
temperature .4 with sampling. Offline frozen SFT comparison: greedy, the same
128 goal IDs, plus200 validation states with newly generated R and action.
Never equate these two decoding protocols. Report native success, mean raw score,
reasoning/strict protocol rates, terminal purchase frequency and marker decisions
/all active decisions. A source-Bad generation label is not a label for fresh R;
assessing route precision/recall needs a separate blinded semantic audit.

No claim of perfect labels, guaranteed reasoning preservation, 1.5B measured
performance, or a newly completed stage-2 run follows from source/CPU tests.
