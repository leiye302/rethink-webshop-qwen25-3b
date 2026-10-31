"""In-process, state-preserving WebShop evaluation; never starts a GPU server.

The CPU native environment lives in a separate Python/Java environment.  Only
literal visible prompts and native transitions cross the JSONL worker boundary.
No source-R prefix, route target, correction, teacher, or hidden goal is supplied
to generation.  Source-Good/Bad groups are provenance groups, not fresh labels.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import queue
import random
import re
import subprocess
import threading
import time
import traceback

import numpy as np
import torch

from sft_reference.route_bce.core import losses
from webshop_data.common import action_admissible, digest, parse_response
from deadline_policy import EvaluationDeadline, check_evaluation


UPSTREAM_COMMIT = "d511f043afe9199b8576fc99ea52e12d84841f32"
MAX_PROMPT_TOKENS = 4096
MAX_NEW_TOKENS = 512
MAX_ENV_STEPS = 15
NATIVE_TASKS = 140
MARKER = "<RETHINK>"


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    with pending.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(pending, path)


class _AtomicRows:
    """Keep recoverable evidence on failure, publish JSONL atomically on success."""
    def __init__(self, path):
        self.path = Path(path)
        self.pending = self.path.with_name(self.path.name + ".pending")
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.pending.open("x", encoding="utf-8")
        return self

    def write(self, row):
        self.handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    def flush(self):
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def __exit__(self, kind, value, tb):
        try:
            self.flush()
        finally:
            self.handle.close()
        if kind is None:
            os.replace(self.pending, self.path)


def _read_rows(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha_file(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _rooted(root, value):
    value = Path(value)
    return value if value.is_absolute() else root / value


@contextmanager
def preserved_evaluation_state(model, tokenizer):
    """Do not change optimizer, parameter gradients, offload hooks, or RNG state.

    Checkpointing booleans are toggled directly: HF enable/disable methods may
    install/remove input hooks, which must remain owned by the training engine.
    """
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    modules = list(model.modules())
    training = [(module, module.training) for module in modules]
    checkpointing = [(module, module.gradient_checkpointing) for module in modules
                     if hasattr(module, "gradient_checkpointing")]
    model_config = getattr(model, "config", None)
    had_cache_flag = model_config is not None and hasattr(model_config, "use_cache")
    cache_flag = model_config.use_cache if had_cache_flag else None
    padding_side = tokenizer.padding_side
    pad_token_id = tokenizer.pad_token_id
    try:
        model.eval()
        for module, _ in checkpointing:
            module.gradient_checkpointing = False
        with torch.inference_mode():
            yield
    finally:
        tokenizer.padding_side = padding_side
        tokenizer.pad_token_id = pad_token_id
        for module, value in checkpointing:
            module.gradient_checkpointing = value
        if had_cache_flag:
            model_config.use_cache = cache_flag
        # Restore every submodule's exact previous mode, including mixed modes.
        for module, value in training:
            module.training = value
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def _eos_ids(model):
    value = model.generation_config.eos_token_id
    values = [value] if isinstance(value, int) else list(value or [])
    if not values or any(not isinstance(item, int) or item < 0 for item in values):
        raise ValueError("Pinned model generation_config must define valid EOS IDs")
    return values


def _autocast(device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                          enabled=device.type == "cuda")


class PromptBudgetError(ValueError):
    def __init__(self, records):
        self.records = records
        super().__init__("truncation=error: prompt exceeds 4096 tokens: " +
                         ", ".join(f"{r['key']}={r['prompt_token_count']}" for r in records))


def generation_metrics(text, ids, marker_id, eos_ids):
    """WebShop protocol, allowing native angle-bracket navigation buttons."""
    protocol = parse_response(text)
    action = protocol.get("sampled_action")
    grammar = bool(isinstance(action, str) and re.fullmatch(r"(?:search|click)\[[^\r\n]+\]", action, re.I)
                   and not any(tag in action for tag in
                               ("<think>", "</think>", "<action>", "</action>", MARKER, "<|"))
                   and not re.search(r"<[aqcr]>", action))
    marker_count = ids.count(marker_id)
    terminated = bool(ids and ids[-1] in eos_ids)
    return {"reasoning_open": "<think>" in text,
            "reasoning_closed": "</think>" in text,
            "strict_protocol": bool(protocol["strict_protocol"]),
            "marker_token_count": marker_count, "marker_present": marker_count > 0,
            "marker_text_agrees": bool(marker_count) == (MARKER in text),
            "parsed_action": action, "action_grammar_valid": grammar,
            "terminated_with_eos": terminated,
            "truncated": len(ids) >= MAX_NEW_TOKENS and not terminated}


def generate_batch(model, tokenizer, prompts, marker, device, benchmark=False):
    """One greedy batch from the actual empty-assistant chat template.

    prompts is a list of (evidence_key, literal_visible_H).  Padding is explicit
    and left-sided; no tokenizer truncation or assistant prefill is possible.
    """
    eos = _eos_ids(model)
    encoded = []
    for key, prompt in prompts:
        ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                            tokenize=True, add_generation_prompt=True)
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if not isinstance(ids, list) or not ids or not isinstance(ids[0], int):
            raise ValueError("Expected nonempty unbatched chat-template token IDs")
        encoded.append({"key": key, "prompt": prompt, "prompt_token_ids": ids,
                        "prompt_token_count": len(ids)})
    overlength = [item for item in encoded if item["prompt_token_count"] > MAX_PROMPT_TOKENS]
    if overlength:
        raise PromptBudgetError(overlength)
    if not encoded:
        return []
    old_side, old_pad = tokenizer.padding_side, tokenizer.pad_token_id
    try:
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = eos[0]
        pad = int(tokenizer.pad_token_id)
        width = max(item["prompt_token_count"] for item in encoded)
        input_ids = torch.full((len(encoded), width), pad, dtype=torch.long, device=device)
        attention = torch.zeros_like(input_ids)
        for index, item in enumerate(encoded):
            length = item["prompt_token_count"]
            input_ids[index, -length:] = torch.tensor(item["prompt_token_ids"], device=device)
            attention[index, -length:] = 1
        sampling = dict(temperature=.4, top_p=1., top_k=0, repetition_penalty=1.) if benchmark else {}
        with _autocast(device):
            result = model.generate(input_ids=input_ids, attention_mask=attention,
                                    do_sample=benchmark, num_beams=1, num_return_sequences=1, **sampling,
                                    max_new_tokens=MAX_NEW_TOKENS, eos_token_id=eos,
                                    pad_token_id=pad, use_cache=True,
                                    return_dict_in_generate=True, output_scores=False)
        sequences = result.sequences if hasattr(result, "sequences") else result
        raw_suffixes = sequences[:, width:].detach().cpu().tolist()
        if len(raw_suffixes) != len(encoded):
            raise ValueError("Generation batch cardinality changed")
        generated = []
        for item, padded in zip(encoded, raw_suffixes):
            # HF pads already-finished rows; preserve that exact tensor separately.
            end = next((index + 1 for index, token in enumerate(padded) if token in eos), len(padded))
            ids = padded[:end]
            text = tokenizer.decode(ids, skip_special_tokens=False,
                                    clean_up_tokenization_spaces=False)
            generated.append({**item, "assistant_prefill": "", "generated_token_ids": ids,
                              "raw_padded_generated_token_ids": padded, "raw_response": text,
                              "generated_token_count": len(ids),
                              "metrics": generation_metrics(text, ids, marker, eos)})
        return generated
    finally:
        tokenizer.padding_side = old_side
        tokenizer.pad_token_id = old_pad


def _group_metrics(records):
    names = ("reasoning_open", "reasoning_closed", "strict_protocol", "marker_present",
             "marker_text_agrees", "action_grammar_valid", "terminated_with_eos", "truncated")
    result = {"count": len(records)}
    for name in names:
        count = sum(bool(row["metrics"][name]) for row in records)
        result[name + "_count"] = count
        result[name + "_rate"] = count / len(records) if records else None
    result["marker_tokens"] = sum(row["metrics"]["marker_token_count"] for row in records)
    result["mean_generated_tokens"] = (sum(row["generated_token_count"] for row in records) /
                                       len(records)) if records else None
    return result


def evaluate_losses(model, valenc, marker, cfg, output, device):
    values = []
    for row in valenc:
        check_evaluation(cfg)
        ids = torch.tensor(row["input_ids"], dtype=torch.long, device=device)
        with _autocast(device):
            result = model(input_ids=ids.unsqueeze(0), attention_mask=torch.ones_like(ids).unsqueeze(0),
                           use_cache=False)
            parts = losses(result.logits[0], ids, int(row["suffix_start"]),
                           int(row["route_label"]), marker, route_lambda=0.1)
        value = {"audit_id": row["audit_id"], "source_route": "bad" if row["route_label"] else "good"}
        value.update({name: float(parts[name].detach().cpu()) for name in
                      ("loss", "suffix_ce", "route_bce", "marker_probability", "binary_logit")})
        if not all(np.isfinite(value[name]) for name in
                   ("loss", "suffix_ce", "route_bce", "marker_probability", "binary_logit")):
            raise FloatingPointError("Non-finite validation loss: " + row["audit_id"])
        values.append(value)
        del result, parts, ids
    summary = {"aggregation": "sample-mean; each row first averages its supervised suffix CE",
               "route_bce_lambda": 0.1, "count": len(values), "groups": {}}
    for group in ("all", "good", "bad"):
        selected = values if group == "all" else [row for row in values if row["source_route"] == group]
        summary["groups"][group] = {"count": len(selected), **{
            name: sum(row[name] for row in selected) / len(selected) if selected else None
            for name in ("loss", "suffix_ce", "route_bce", "marker_probability", "binary_logit")}}
    with _AtomicRows(Path(output) / "loss_rows.jsonl") as evidence:
        for row in values:
            evidence.write(row)
    _atomic_json(Path(output) / "loss_metrics.json", summary)
    return summary


def evaluate_generation(model, tokenizer, rows, marker, cfg, output, device):
    generated = []
    batch_size = int(cfg.get("generation_batch_size", 8))
    if not 1 <= batch_size <= 16:
        raise ValueError("generation_batch_size must be between 1 and 16")
    with _AtomicRows(Path(output) / "generation_val200.jsonl") as evidence:
        for start in range(0, len(rows), batch_size):
            check_evaluation(cfg)
            source = rows[start:start + batch_size]
            batch = generate_batch(model, tokenizer,
                                   [(row["audit_id"], row["prompt"]) for row in source], marker, device)
            for original, record in zip(source, batch):
                record.update({"audit_id": original["audit_id"], "task_id": original["task_id"],
                               "source_route": original["route"],
                               "source_route_is_generated_reasoning_label": False})
                evidence.write(record)
                generated.append(record)
            evidence.flush()
    summary = {"count": len(generated), "assistant_prefill": "", "do_sample": False,
               "max_new_tokens": MAX_NEW_TOKENS, "skip_special_tokens": False,
               "source_group_interpretation": "Stored H/source-R provenance only; NOT semantic labels, "
                                              "precision, or recall of the newly generated reasoning.",
               "overall": _group_metrics(generated),
               "source_good": _group_metrics([row for row in generated if row["source_route"] == "good"]),
               "source_bad": _group_metrics([row for row in generated if row["source_route"] == "bad"])}
    _atomic_json(Path(output) / "generation_metrics.json", summary)
    return summary


def fixed_native_tasks(path):
    from rethink_webshop.fixed_validation import manifest
    fixed = manifest()
    expected = {"schema": "webshop-shared140-native-eval-v1", "task_count": NATIVE_TASKS,
                "selection": fixed["selection"],
                "task_ids": [row["task_id"] for row in fixed["selection"]]}
    expected["tasks_sha256"] = digest(expected["task_ids"])
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise ValueError("Fixed native evaluation tasks changed; start a new shared140 run")
    else:
        _atomic_json(path, expected)
    return expected


class NativeWorker:
    """A bounded JSONL client; CPU environment cannot import into GPU Python."""
    def __init__(self, cfg, output):
        root = Path(cfg["root"]).resolve()
        cpu_env = _rooted(root, cfg.get("native_env_path", "env-webshop"))
        python = _rooted(root, cfg.get("cpu_python", cfg.get("native_python", str(cpu_env / "bin/python"))))
        java = _rooted(root, cfg.get("java_home", str(cpu_env)))
        worker = Path(__file__).with_name("native_worker.py")
        self.timeout = float(cfg.get("native_worker_timeout_seconds", 600))
        self.process = None
        self.log = None
        self.serial = 0
        self.messages = queue.Queue()
        self.thread = None
        self.launch = {"python": str(python), "worker": str(worker), "JAVA_HOME": str(java),
                       "JVM_PATH": str(java / "lib/server/libjvm.so"),
                       "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(root / "src")}
        env = os.environ.copy()
        env.update({"CUDA_VISIBLE_DEVICES": "", "JAVA_HOME": str(java),
                    "JVM_PATH": str(java / "lib/server/libjvm.so"),
                    "PYTHONPATH": str(root / "src"), "PYTHONUNBUFFERED": "1"})
        env["PATH"] = str(java / "bin") + os.pathsep + str(cpu_env / "bin") + os.pathsep + env.get("PATH", "")
        command = [str(python), "-u", str(worker), "--root", str(root)]
        try:
            self.log = (Path(output) / "native_worker.stderr.log").open("x", encoding="utf-8")
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=self.log, text=True, encoding="utf-8", bufsize=1,
                                            cwd=root, env=env)
            self.thread = threading.Thread(target=self._reader, daemon=True)
            self.thread.start()
            self.ready = self._receive()
            if not self.ready.get("ok") or self.ready.get("event") != "ready":
                raise RuntimeError("Native worker failed to initialize: " + json.dumps(self.ready))
        except BaseException:
            self.close(graceful=False)
            raise

    def _reader(self):
        try:
            for line in self.process.stdout:
                self.messages.put(line)
        finally:
            self.messages.put(None)

    def _receive(self):
        try:
            line = self.messages.get(timeout=self.timeout)
        except queue.Empty:
            raise TimeoutError("Native worker response timed out; see native_worker.stderr.log") from None
        if line is None:
            raise RuntimeError("Native worker exited before response; see native_worker.stderr.log")
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            raise RuntimeError("Non-JSON native worker stdout: " + line[:500]) from None

    def request(self, operation, **payload):
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("Native CPU worker is not alive")
        self.serial += 1
        request_id = self.serial
        self.process.stdin.write(json.dumps({"request_id": request_id, "op": operation, **payload},
                                           ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        response = self._receive()
        if response.get("request_id") != request_id or not response.get("ok"):
            raise RuntimeError("Native worker request failed: " + json.dumps(response, ensure_ascii=False))
        return response

    def close(self, graceful=True):
        process = self.process
        try:
            if process is not None:
                if process.poll() is None and graceful:
                    try:
                        self.request("close")
                    except Exception:
                        pass
                if process.stdin is not None:
                    process.stdin.close()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
                if self.thread is not None:
                    self.thread.join(timeout=5)
                if process.stdout is not None:
                    process.stdout.close()
        finally:
            if self.log is not None:
                self.log.close()
            self.process = None


def evaluate_native(model, tokenizer, marker, cfg, step, output, device, tasks):
    output = Path(output)
    concurrency = int(cfg.get("native_concurrency", 16))
    batch_size = int(cfg.get("generation_batch_size", 8))
    if concurrency not in (8, 16) or not 1 <= batch_size <= 16:
        raise ValueError("Native concurrency must be 8 or 16; generation batch size 1..16")
    protocol = {"upstream_commit": UPSTREAM_COMMIT,
                "benchmark": "SDAR native WebShop; 1000 products and synthetic goals, NOT full human WebShop",
                "shared_evaluation_contract": "configs/evaluation_protocol.json",
                "task_selection": tasks, "rollouts_per_task": 1, "history_length": 2,
                "environment_seeds": list(range(1000, 1140)), "max_steps": MAX_ENV_STEPS,
                "max_prompt_tokens": MAX_PROMPT_TOKENS, "prompt_truncation": "error",
                "max_new_tokens": MAX_NEW_TOKENS, "do_sample": True, "temperature": .4, "top_p": 1., "top_k": -1,
                "assistant_prefill": "", "skip_special_tokens": False,
                "native_projection_unchanged": True, "native_rewards_unchanged": True,
                "forced_route": False, "retries": 0, "teacher_or_hint": False,
                "success_definition": "native done and raw_task_score == 1.0",
                "rule_reward": "10 if successful else 0", "concurrent_episodes": concurrency,
                "generation_batch_size": batch_size}
    episodes, metrics = [], []
    selections = {row["task_id"]: row for row in tasks["selection"]}
    initial_observations = {}
    pending = deque(tasks["task_ids"])
    active = {}
    worker = None
    try:
        worker = NativeWorker(cfg, output)
        identity = worker.ready["identity"]
        if identity["products"] != 1000 or identity["human_goals"] or not identity["use_small"]:
            raise ValueError("Native environment is not the agreed 1000-product synthetic-goal setup")
        if identity["upstream_commit"] != UPSTREAM_COMMIT or identity["environment_seed"] != 1000:
            raise ValueError("Native environment seed/commit changed")
        manifest = {"protocol": protocol, "protocol_sha256": digest(protocol),
                    "environment": identity, "native_sources": worker.ready["native_sources"],
                    "worker_launch": worker.launch, "step": int(step)}
        _atomic_json(output / "native_manifest.json", manifest)
        with _AtomicRows(output / "native_decisions.jsonl") as decisions, \
                _AtomicRows(output / "native_episodes.jsonl") as episode_evidence, \
                _AtomicRows(output / "native_generations.jsonl") as generation_evidence:
            while pending or active:
                check_evaluation(cfg)
                reset_items = []
                while pending and len(active) + len(reset_items) < concurrency:
                    task_id = pending.popleft()
                    selected = selections[task_id]
                    session = f"eval{int(step):06d}_goal{selected['goal_index']:05d}"
                    reset_items.append({"session_id": session, "task_id": task_id,
                                        "goal_index": selected["goal_index"], "worker_seed": selected["worker_seed"]})
                if reset_items:
                    response = worker.request("reset", items=reset_items)
                    if len(response["items"]) != len(reset_items):
                        raise ValueError("Native reset cardinality changed")
                    for expected, item in zip(reset_items, response["items"]):
                        if item["session_id"] != expected["session_id"] or item["task_id"] != expected["task_id"]:
                            raise ValueError("Native reset order/identity changed")
                        initial_observations[item["task_id"]] = item["state"]["observation"]
                        active[item["session_id"]] = {**item, "steps": 0, "executed_actions": []}
                # Process every live episode once before refilling.  Completion
                # never selects a replacement rollout of the same goal.
                live = list(active)
                for start in range(0, len(live), batch_size):
                    check_evaluation(cfg)
                    sessions = live[start:start + batch_size]
                    prompts = [(session + f"_s{active[session]['steps']:02d}",
                                active[session]["state"]["prompt"]) for session in sessions]
                    try:
                        generated = generate_batch(model, tokenizer, prompts, marker, device, benchmark=True)
                    except PromptBudgetError as error:
                        for item in error.records:
                            session = next(s for s in sessions if item["key"].startswith(s + "_s"))
                            decisions.write({"status": "blocked_prompt", **item,
                                             "task_id": active[session]["task_id"],
                                             "session_id": session, "state": active[session]["state"]})
                        raise
                    step_items = [{"session_id": session, "response": record["raw_response"],
                                   "expected_state_sha256": active[session]["state"]["state_sha256"]}
                                  for session, record in zip(sessions, generated)]
                    # Commit literal generations before any native action.  A
                    # worker failure never erases the policy's attempted output.
                    for session, record in zip(sessions, generated):
                        generation_evidence.write({"session_id": session,
                            "task_id": active[session]["task_id"], "step": active[session]["steps"],
                            "state": active[session]["state"], "generation": record})
                    generation_evidence.flush()
                    results = worker.request("step", items=step_items)["items"]
                    if len(results) != len(sessions):
                        raise ValueError("Native transition cardinality changed")
                    release = []
                    for session, generated_row, transition in zip(sessions, generated, results):
                        if transition["session_id"] != session:
                            raise ValueError("Native transition order changed")
                        episode = active[session]
                        before = episode["state"]
                        native = transition["result"]
                        if not 0 <= native["raw_task_score"] <= 1:
                            raise ValueError("Native score outside [0,1]")
                        legal = action_admissible(native["environment_action"], before["available_actions"])
                        record = {"session_id": session, "task_id": episode["task_id"],
                                  "task_sha256": episode["task_sha256"], "rollout_id": 0,
                                  "step": episode["steps"], "state": before,
                                  "executed_prefix": list(episode["executed_actions"]),
                                  "generation": generated_row, **native,
                                  "executed_action_admissible": legal,
                                  "projection_used_fallback": transition["projection_used_fallback"],
                                  "next_state": transition["next_state"]}
                        decisions.write(record)
                        metrics.append({**generated_row, "native_format_valid": native["native_format_valid"],
                                        "executed_action_admissible": legal,
                                        "projection_used_fallback": transition["projection_used_fallback"]})
                        episode["steps"] += 1
                        episode["executed_actions"].append(native["environment_action"])
                        if native["done"] or episode["steps"] >= MAX_ENV_STEPS:
                            final = {"session_id": session, "task_id": episode["task_id"],
                                     "task_sha256": episode["task_sha256"], "rollout_id": 0,
                                     "steps": episode["steps"], "done": native["done"],
                                     "raw_task_score": native["raw_task_score"],
                                     "success": bool(native["done"] and native["raw_task_score"] == 1.0),
                                     "training_reward": native["training_reward"],
                                     "stop_reason": "environment_done" if native["done"] else "step_limit"}
                            episode_evidence.write(final)
                            episodes.append(final)
                            release.append(session)
                        else:
                            if transition["next_state"] is None:
                                raise ValueError("Nonterminal native step has no visible next state")
                            episode["state"] = transition["next_state"]
                    if release:
                        worker.request("close", session_ids=release)
                        for session in release:
                            del active[session]
                    decisions.flush()
                    episode_evidence.flush()
                _atomic_json(output / "native_progress.json", {"completed_episodes": len(episodes),
                             "target_episodes": NATIVE_TASKS, "decisions": len(metrics),
                             "tasks_sha256": tasks["tasks_sha256"], "updated_unix": time.time()})
        if len(episodes) != NATIVE_TASKS or {e["task_id"] for e in episodes} != set(tasks["task_ids"]):
            raise ValueError("Native evaluation must have exactly one episode for each fixed goal")
        from rethink_webshop.fixed_validation import digest as fixed_digest, record_identity
        from evaluation_contract import MANIFEST_SHA256, SPLIT_SHA256
        fixed_identity = {"protocol": "webshop-shared140-v1",
            "manifest_sha256": MANIFEST_SHA256, "task_split_sha256": SPLIT_SHA256,
            "task_ids": tasks["task_ids"], "selection_sha256": fixed_digest(tasks["selection"]),
            "initial_observations_sha256": fixed_digest([initial_observations[k] for k in tasks["task_ids"]])}
        from rethink_webshop.fixed_validation import manifest as fixed_manifest
        fixed_identity["protocol"] = fixed_manifest()["protocol"]
        record_identity(output.parent, fixed_identity)
        summary = {"fixed_validation_identity": fixed_identity, "count": len(episodes), "decisions": len(metrics),
                   "raw_task_score_mean": sum(row["raw_task_score"] for row in episodes) / len(episodes),
                   "success_count": sum(row["success"] for row in episodes),
                   "success_rate": sum(row["success"] for row in episodes) / len(episodes),
                   "training_reward_mean": sum(row["training_reward"] for row in episodes) / len(episodes),
                   "done_count": sum(row["done"] for row in episodes),
                   "step_limit_count": sum(row["stop_reason"] == "step_limit" for row in episodes),
                   "decision_metrics": _group_metrics(metrics), "tasks_sha256": tasks["tasks_sha256"],
                   "protocol_sha256": digest(protocol), "environment": identity}
        for name in ("native_format_valid", "executed_action_admissible", "projection_used_fallback"):
            count = sum(bool(row[name]) for row in metrics)
            summary["decision_metrics"][name + "_count"] = count
            summary["decision_metrics"][name + "_rate"] = count / len(metrics) if metrics else None
        _atomic_json(output / "native_metrics.json", summary)
        return summary
    except EvaluationDeadline as error:
        _atomic_json(output / "native_deferred.json", {"status":"deferred_shutdown_budget",
                     "completed_episodes":len(episodes),"decisions":len(metrics),
                     "tasks_sha256":tasks['tasks_sha256'],"message":str(error)})
        raise
    except BaseException as error:
        _atomic_json(output / "native_error.json", {"error_type": type(error).__name__, "message": str(error),
                     "completed_episodes": len(episodes), "decisions": len(metrics),
                     "tasks_sha256": tasks["tasks_sha256"], "traceback": traceback.format_exc()})
        raise
    finally:
        if worker is not None:
            worker.close()


def run_suite(model, tokenizer, rows, valenc, marker, cfg, step, output):
    """Full 902-row loss, fixed 200-H generation, and shared140 native episodes (T=0.4); greedy200 is a separate diagnostic."""
    if int(step) not in (0, 750, 1500):
        raise ValueError("Frozen evaluation checkpoints are 0, 750, and 1500")
    if len(rows) != 902 or len(valenc) != 902:
        raise ValueError("Full frozen validation must contain 902 rows")
    if {route: sum(row["route"] == route for row in rows) for route in ("good", "bad")} != {"good": 631, "bad": 271}:
        raise ValueError("Validation source-group counts changed")
    for row, encoded in zip(rows, valenc):
        if row["audit_id"] != encoded["audit_id"] or int(row["route"] == "bad") != encoded["route_label"]:
            raise ValueError("Validation raw/encoded row identity mismatch")
    if float(cfg.get("route_bce_lambda", 0.1)) != 0.1:
        raise ValueError("Frozen evaluation uses route_bce_lambda=0.1")
    root = Path(cfg["root"]).resolve()
    dataset = _rooted(root, cfg.get("dataset_path", "datasets/g7500_b4500_ep1_v1"))
    generation_path = _rooted(root, cfg["generation_rows_path"]) if cfg.get("generation_rows_path") else dataset / "generation_val200.jsonl"
    generation_rows = _read_rows(generation_path)
    if len(generation_rows) != 200 or {route: sum(row["route"] == route for row in generation_rows)
                                      for route in ("good", "bad")} != {"good": 100, "bad": 100}:
        raise ValueError("Fixed H generation set must contain 100 source-Good and 100 source-Bad")
    val_by_id = {row["audit_id"]: row for row in rows}
    if len(val_by_id) != 902 or len({row["audit_id"] for row in generation_rows}) != 200:
        raise ValueError("Duplicate validation/generation decision identity")
    if any(row != val_by_id.get(row["audit_id"]) for row in generation_rows):
        raise ValueError("Fixed generation rows are not exact members of the frozen validation")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "suite.json").exists():
        raise FileExistsError("Refusing an implicit evaluation retry/overwrite: " + str(output))
    tasks = fixed_native_tasks(output.parent / "native_tasks.json")
    device = torch.device(cfg.get("device", "cuda:0"))
    started = time.time()
    summary = {"schema": "webshop-sft-evaluation-v1", "step": int(step), "status": "running",
               "started_unix": started, "generation_rows_sha256": _sha_file(generation_path),
               "generation_ids_sha256": digest([row["audit_id"] for row in generation_rows]),
               "validation_ids_sha256": digest([row["audit_id"] for row in rows]),
               "tasks_sha256": tasks["tasks_sha256"], "model_device": str(device),
               "source_group_semantics": "source-R label only; generated reasoning is not semantically relabeled"}
    _atomic_json(output / "suite.json", summary)
    stage = "loss"
    try:
        with preserved_evaluation_state(model, tokenizer):
            summary["loss"] = evaluate_losses(model, valenc, marker, cfg, output, device)
            stage = "generation"
            summary["generation"] = evaluate_generation(model, tokenizer, generation_rows, marker, cfg, output, device)
            stage = "native_environment"
            summary["native"] = evaluate_native(model, tokenizer, marker, cfg, step, output, device, tasks)
        summary["status"] = "complete"
    except EvaluationDeadline as error:
        summary.update({"status":"deferred_shutdown_budget","deferred_stage":stage,
                        "completed":False,"reason":str(error)})
        _atomic_json(output / "suite_deferred.json", summary)
        return summary
    except BaseException as error:
        summary.update({"status": "failed", "failed_stage": stage,
                        "error_type": type(error).__name__, "error": str(error)})
        raise
    finally:
        summary["elapsed_seconds"] = time.time() - started
        summary["finished_unix"] = time.time()
        _atomic_json(output / "suite.json", summary)
    # This is the training engine's acceptance marker.  It is published only
    # after environment teardown, state restoration, and every report write.
    _atomic_json(output / "suite_complete.json", summary)
    return summary
