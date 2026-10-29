"""Guarded, full-parameter WebShop SFT; importing this module never starts work.

The frozen reference owns tokenization, sampling, and the loss.  This file only
owns hardware plumbing, receipts, evaluation boundaries, and exact checkpoints.
Preflight updates a disposable model twice and NEVER writes its weights.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time
import uuid


PARENT_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
EXPECTED_ROOT = str(Path(__file__).resolve().parents[2])
DEFAULTS = {
    "model_path": "assets/base",
    "dataset_path": "datasets/frozen_sft",
    "output_path": "runs/sft",
    "receipts_path": "receipts/webshop_sft",
    "reference_path": "src/sft_reference",
    "reference_lock_path": "configs/reference.lock.json",
    "model_lock_path": "configs/model.lock.json",
    "max_length": 8192,
    "seed": 0,
    "epochs": 1,
    "total_updates": 1500,
    "global_batch": 8,
    "micro_batch": 1,
    "good_per_batch": 5,
    "bad_per_batch": 3,
    "learning_rate": 2e-6,
    "betas": [0.9, 0.95],
    "weight_decay": 0.01,
    "adam_epsilon": 1e-8,
    "gradient_clip": 1.0,
    "route_bce_lambda": 0.1,
    "warmup_steps": 150,
    "checkpoint_steps": [750, 1500],
    "evaluation_steps": [750, 1500],
    "cpu_threads": 16,
    "optimizer_backend": "native_torch_adamw_cpu_fp32_offload",
    "training_authorized": False,
}


def emit(event, **payload):
    print(json.dumps({"event": event, **payload}, ensure_ascii=False, allow_nan=False), flush=True)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending-" + uuid.uuid4().hex)
    with pending.open("x", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(pending, path)


def within(root, path):
    root = Path(root).resolve()
    result = (root / path).resolve()
    if not result.is_relative_to(root) or result == root:
        raise ValueError("Project-relative path escapes or equals the project root")
    return result


def load_config(path):
    supplied = read_json(path)
    if not isinstance(supplied, dict):
        raise ValueError("Configuration must be a JSON object")
    cfg = {**DEFAULTS, **supplied}
    # These are named experimental invariants, not tuning defaults.
    for key in ("seed", "epochs", "total_updates", "global_batch", "micro_batch",
                "good_per_batch", "bad_per_batch", "learning_rate", "betas",
                "weight_decay", "adam_epsilon", "gradient_clip", "route_bce_lambda",
                "warmup_steps", "checkpoint_steps", "evaluation_steps"):
        if cfg[key] != DEFAULTS[key]:
            raise ValueError("Frozen experiment configuration differs: " + key)
    if cfg.get("model_revision", PARENT_REVISION) != PARENT_REVISION:
        raise ValueError("Wrong model revision")
    if cfg.get("sft_reference_kl", 0) != 0:
        raise ValueError("No reference KL is authorized")
    if any(cfg.get(key) for key in ("lora", "qlora", "quantization", "teacher", "opsd")):
        raise ValueError("Only full-parameter non-quantized SFT is authorized")
    if not isinstance(cfg["max_length"], int) or not 0 < cfg["max_length"] <= 8192:
        raise ValueError("max_length must be in [1,8192]; overlength rows are errors")
    if not isinstance(cfg["cpu_threads"], int) or cfg["cpu_threads"] < 1:
        raise ValueError("cpu_threads must be positive")
    if cfg["optimizer_backend"] not in ("native_torch_adamw_cpu_fp32_offload", "native_torch_adamw_cuda_fp32"):
        raise ValueError("Only native full-precision AdamW CPU-offload or GPU-resident backends are allowed")
    return cfg


def verify_reference(root, cfg):
    directory = within(root, cfg["reference_path"])
    lock_path = within(root, cfg["reference_lock_path"])
    lock = read_json(lock_path)
    required = {"route_bce/core.py", "route_bce/__init__.py",
                "legacy/stage1_training.py", "legacy/rethink_token.py"}
    if set(lock["files"]) != required:
        raise ValueError("Unexpected frozen reference member set")
    for name, expected in lock["files"].items():
        if sha(within(directory, name)) != expected:
            raise ValueError("Frozen reference checksum differs: " + name)
    sys.path.insert(0, str(directory))
    core = importlib.import_module("route_bce.core")
    if Path(core.__file__).resolve() != directory / "route_bce/core.py":
        raise ValueError("An unrelated route_bce package shadows the frozen reference")
    return core, {"lock_sha256": sha(lock_path), "files": lock["files"]}


def verify_parent(root, cfg):
    directory = within(root, cfg["model_path"])
    lock_path = within(root, cfg["model_lock_path"])
    lock, receipt = read_json(lock_path), read_json(directory / "verified_model.json")
    if lock["revision"] != PARENT_REVISION or receipt["revision"] != PARENT_REVISION:
        raise ValueError("Unverified parent model revision")
    if set(lock["files"]) != set(receipt["files"]):
        raise ValueError("Parent receipt member set differs from the pinned lock")
    for name, spec in lock["files"].items():
        target = within(directory, name)
        actual = receipt["files"][name]
        if target.stat().st_size != spec["bytes"] or actual["bytes"] != spec["bytes"]:
            raise ValueError("Partial parent model file: " + name)
        checksum = sha(target)
        if checksum != actual["sha256"] or (spec.get("sha256") and checksum != spec["sha256"]):
            raise ValueError("Parent per-file SHA256 differs: " + name)
        if not spec.get("sha256"):
            data = target.read_bytes()
            blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            if blob != spec["git_blob"]:
                raise ValueError("Parent metadata Git blob differs: " + name)
    return {"revision": PARENT_REVISION, "verified_model_sha256": sha(directory / "verified_model.json"),
            "model_lock_sha256": sha(lock_path), "files": receipt["files"]}


def validate_dataset(root, cfg, core):
    directory = within(root, cfg["dataset_path"])
    manifest_path = directory / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("user_authorized_repartition") is not True:
        raise ValueError("Whole-task repartition authorization is missing")
    for name, spec in manifest["files"].items():
        path = within(directory, name)
        if path.stat().st_size != spec["bytes"] or sha(path) != spec["sha256"]:
            raise ValueError("Frozen dataset member differs: " + name)
    train = core.load_rows(directory / "train.jsonl")
    val = core.load_rows(directory / "val.jsonl")
    generation = core.load_rows(directory / "generation_val200.jsonl")
    expected = ((train, 7500, 4500), (val, 631, 271), (generation, 100, 100))
    for rows, good, bad in expected:
        if len(rows) != good + bad or sum(r["route"] == "good" for r in rows) != good \
                or sum(r["route"] == "bad" for r in rows) != bad:
            raise ValueError("Dataset route counts differ from the frozen recipe")
        ids = [r["audit_id"] for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError("Repeated audit_id in the dataset")
    if set(r["audit_id"] for r in train) & set(r["audit_id"] for r in val):
        raise ValueError("Training/validation row overlap")
    validate_task_separation(train, val)
    if not set(r["audit_id"] for r in generation) <= set(r["audit_id"] for r in val):
        raise ValueError("Generation subset is not fixed inside validation")
    order = core.batches(train, seed=cfg["seed"], epoch=0)
    validate_order(train, order, cfg["total_updates"])
    identity = {"manifest_sha256": sha(manifest_path), "files": manifest["files"],
                "train_rows": len(train), "validation_rows": len(val),
                "generation_rows": len(generation), "order_sha256": digest(order)}
    return train, val, order, identity


def validate_task_separation(train, val):
    for field in ("task_id", "task_sha256"):
        if any(field not in r or r[field] is None for r in train + val):
            raise ValueError("Task identity is missing: " + field)
        if {r[field] for r in train} & {r[field] for r in val}:
            raise ValueError("Training/validation task overlap: " + field)


def validate_order(rows, order, expected_updates):
    flattened = [index for batch in order for index in batch]
    if len(order) != expected_updates or len(flattened) != len(rows) \
            or sorted(flattened) != list(range(len(rows))):
        raise ValueError("One epoch must visit every row once without dropping/cycling")
    for batch in order:
        if len(batch) != 8 or sum(rows[i]["route"] == "good" for i in batch) != 5 \
                or sum(rows[i]["route"] == "bad" for i in batch) != 3:
            raise ValueError("Every global batch must contain exactly 5 Good and 3 Bad")


def encode_all(rows, tokenizer, core, max_length):
    result = []
    for row in rows:
        encoded = core.encode_row(row, tokenizer, max_length=max_length)
        if encoded["joint_tokenization_matches"] is not True:
            raise ValueError("Joint tokenization differs at row " + encoded["audit_id"])
        result.append(encoded)
    return result


def code_identity(root, cfg):
    sources = sorted(Path(__file__).resolve().parent.glob("*.py"))
    sources += [within(root, cfg["reference_lock_path"]), within(root, cfg["model_lock_path"])]
    source_root = within(root, cfg["reference_path"])
    sources += [source_root / name for name in read_json(within(root, cfg["reference_lock_path"]))["files"]]
    result = {}
    for path in sources:
        if not path.is_relative_to(Path(root).resolve()):
            raise ValueError("Training source must be deployed inside this project")
        result[str(path.relative_to(root))] = sha(path)
    return result


def runtime_check(cfg):
    import torch
    import transformers
    if sys.version_info[:2] != (3, 12) or torch.__version__.split("+")[0] != "2.8.0" \
            or torch.version.cuda != "12.8" or transformers.__version__ != "4.57.1":
        raise RuntimeError("Expected Python3.12/torch2.8.0/CUDA12.8/transformers4.57.1")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one authorized idle CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU does not support BF16 autocast")
    torch.set_num_threads(cfg["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    transformers.logging.set_verbosity_error()
    transformers.utils.logging.disable_progress_bar()
    p = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info(0)
    return {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "cuda": torch.version.cuda,
            "gpu_name": p.name, "gpu_total_bytes": total, "gpu_free_before_load_bytes": free,
            "cpu_threads": cfg["cpu_threads"], "visible_gpu_count": 1}


def rng_state():
    import torch
    return {"python": random.getstate(), "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    import torch
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


@contextmanager
def preserve_rng():
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)


def seed_all(seed):
    import torch
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_scheduler(optimizer, warmup=150, total=1500):
    import torch
    def factor(step):
        if step < warmup:
            return float(step) / max(1, warmup)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / (total - warmup))))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def memory_status():
    result = {}
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            key, _, value = line.partition(":")
            if key in ("VmRSS", "VmHWM"):
                result[{"VmRSS": "process_rss_bytes", "VmHWM": "process_peak_rss_bytes"}[key]] = int(value.split()[0]) * 1024
    info = Path("/proc/meminfo")
    if info.exists():
        for line in info.read_text().splitlines():
            if line.startswith(("MemAvailable:", "MemTotal:")):
                key, value = line.split(":", 1)
                result[{"MemAvailable": "host_available_bytes", "MemTotal": "host_total_bytes"}[key]] = int(value.split()[0]) * 1024
    return result


def verify_offload_numerics(device="cpu"):
    """Tiny verification of accumulated gradients and a single AdamW update.

    This is deliberately labelled synthetic; it is not the full-model GPU test.
    """
    import copy
    import torch
    from cpu_offload import CpuAdamWOffload
    with preserve_rng():
        torch.manual_seed(941)
        reference = torch.nn.Sequential(torch.nn.Linear(7, 11), torch.nn.GELU(), torch.nn.Linear(11, 5)).to(device)
        model = copy.deepcopy(reference)
        ordinary = torch.optim.AdamW(reference.parameters(), lr=2e-6, betas=(.9, .95),
                                     weight_decay=.01, eps=1e-8, foreach=True)
        offload = CpuAdamWOffload(model)
        initial = [p.detach().clone() for p in model.parameters()]
        xs = [torch.randn(3, 7, device=device) for _ in range(8)]
        ys = [torch.randn(3, 5, device=device) for _ in range(8)]
        ordinary.zero_grad(set_to_none=True)
        offload.begin()
        autocast = device.startswith("cuda")
        max_grad_error = 0.0
        try:
            for x, y in zip(xs, ys):
                with torch.autocast(device_type="cuda" if autocast else "cpu", dtype=torch.bfloat16, enabled=autocast):
                    ref_loss = (reference(x).float() - y).square().mean() / 8
                    test_loss = (model(x).float() - y).square().mean() / 8
                ref_loss.backward()
                test_loss.backward()
                for p, start in zip(model.parameters(), initial):
                    torch.testing.assert_close(p, start, rtol=0, atol=0)
                    if p.grad is not None:
                        raise AssertionError("GPU leaf gradient was not drained")
            for p, master in zip(reference.parameters(), offload.masters):
                max_grad_error = max(max_grad_error, float((p.grad.cpu() - master.grad).abs().max()))
                torch.testing.assert_close(p.grad.cpu(), master.grad, rtol=3e-5, atol=3e-7)
            reference_norm = float(torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0, foreach=True, error_if_nonfinite=True))
            ordinary.step()
            test_norm = offload.finish(clip=1.0)
            max_weight_error = 0.0
            for p, q in zip(reference.parameters(), model.parameters()):
                max_weight_error = max(max_weight_error, float((p - q).abs().max()))
                torch.testing.assert_close(p, q, rtol=1e-5, atol=1e-7)
            if not math.isclose(reference_norm, test_norm, rel_tol=1e-5, abs_tol=1e-6):
                raise AssertionError("Global gradient clipping norms differ")
            return {"passed": True, "synthetic_only": True, "device": device, "micro_samples": 8,
                    "adamw_steps": 1, "max_gradient_abs_error": max_grad_error,
                    "max_parameter_abs_error": max_weight_error, "global_gradient_norm": test_norm}
        finally:
            offload.close()


def make_optimizer(model, cfg):
    if cfg["optimizer_backend"] == "native_torch_adamw_cuda_fp32":
        from gpu_resident import GpuResidentAdamW
        cls = GpuResidentAdamW
    else:
        from cpu_offload import CpuAdamWOffload
        cls = CpuAdamWOffload
    return cls(model, lr=cfg["learning_rate"], betas=tuple(cfg["betas"]),
               weight_decay=cfg["weight_decay"])


def verify_resident_numerics(device="cuda:0"):
    """Tiny native-AdamW parity check, including an exact optimizer-state resume."""
    import copy
    import torch
    from gpu_resident import GpuResidentAdamW
    with preserve_rng():
        torch.manual_seed(941)
        reference = torch.nn.Sequential(torch.nn.Linear(7, 11), torch.nn.GELU(), torch.nn.Linear(11, 5)).to(device)
        model = copy.deepcopy(reference)
        ordinary = torch.optim.AdamW(reference.parameters(), lr=2e-6, betas=(.9, .95),
                                     weight_decay=.01, eps=1e-8, foreach=True)
        resident = GpuResidentAdamW(model)
        xs = [torch.randn(3, 7, device=device) for _ in range(8)]
        ys = [torch.randn(3, 5, device=device) for _ in range(8)]
        max_gradient_error, max_weight_error = 0., 0.
        for step in range(2):
            initial = [p.detach().clone() for p in model.parameters()]
            ordinary.zero_grad(set_to_none=True)
            resident.begin()
            for x, y in zip(xs, ys):
                with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu",
                                    dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                    ref_loss = (reference(x).float() - y).square().mean() / 8
                    test_loss = (model(x).float() - y).square().mean() / 8
                ref_loss.backward()
                test_loss.backward()
                for p, start in zip(model.parameters(), initial):
                    torch.testing.assert_close(p, start, rtol=0, atol=0)
            for p, q in zip(reference.parameters(), model.parameters()):
                max_gradient_error = max(max_gradient_error, float((p.grad-q.grad).detach().abs().max()))
                torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
            reference_norm = float(torch.nn.utils.clip_grad_norm_(reference.parameters(), 1., foreach=True,
                                                                  error_if_nonfinite=True))
            ordinary.step()
            norm = resident.finish()
            if norm != reference_norm:
                raise AssertionError("Native GPU clipping norms differ")
            for p, q in zip(reference.parameters(), model.parameters()):
                max_weight_error = max(max_weight_error, float((p-q).detach().abs().max()))
                torch.testing.assert_close(p, q, rtol=0, atol=0)
            if step == 0:
                restored = GpuResidentAdamW(model)
                restored.optimizer.load_state_dict(copy.deepcopy(resident.optimizer.state_dict()))
                restored.updates = resident.updates
                resident.close()
                resident = restored
        resident.close()
        return {"passed": True, "synthetic_only": True, "device": device, "micro_samples": 8,
                "adamw_steps": 2, "exact_optimizer_state_resume_checked": True,
                "max_gradient_abs_error": max_gradient_error, "max_parameter_abs_error": max_weight_error,
                "global_gradient_norm": norm, "optimizer_backend": "native_torch_adamw_cuda_fp32"}


def load_model(path, tokenizer, core):
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(str(path), local_files_only=True,
        torch_dtype=torch.float32, attn_implementation="sdpa", low_cpu_mem_usage=True)
    marker = core.register_token(tokenizer, model=model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    model.to("cuda:0")
    if not all(p.requires_grad and p.dtype == torch.float32 for p in model.parameters()):
        raise ValueError("Every model parameter must be trainable FP32")
    return model, marker


def global_batch(model, offload, encoded, indices, core, marker, cfg, *, cuda=True):
    import torch
    if len(indices) != 8 or sum(encoded[i]["route_label"] == 0 for i in indices) != 5:
        raise ValueError("Expected one complete 5:3 global batch")
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()
    offload.begin()
    losses = {"loss": 0., "suffix_ce": 0., "route_bce": 0.}
    tokens = 0
    for index in indices:
        row = encoded[index]
        ids = torch.tensor(row["input_ids"], dtype=torch.long, device="cuda:0" if cuda else "cpu")
        with torch.autocast(device_type="cuda" if cuda else "cpu", dtype=torch.bfloat16, enabled=cuda):
            output = model(input_ids=ids.unsqueeze(0), attention_mask=torch.ones_like(ids).unsqueeze(0),
                           use_cache=False, return_dict=True)
            terms = core.losses(output.logits[0], ids, row["suffix_start"], row["route_label"],
                                marker, route_lambda=cfg["route_bce_lambda"])
            scaled = terms["loss"] / len(indices)
        if not bool(torch.isfinite(scaled)):
            raise FloatingPointError("Nonfinite full-model loss")
        scaled.backward()
        for key in losses:
            losses[key] += float(terms[key].detach()) / len(indices)
        tokens += len(row["input_ids"])
        del terms, output, scaled, ids
    if cuda:
        torch.cuda.synchronize()
    update_begin = time.perf_counter()
    gradient_norm = offload.finish(clip=cfg["gradient_clip"])
    if cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - begin
    result = {**losses, "seconds": elapsed, "optimizer_step_seconds": time.perf_counter() - update_begin,
              "optimizer_backend": cfg.get("optimizer_backend", "native_torch_adamw_cpu_fp32_offload"),
              "gradient_norm_before_clip": gradient_norm, "tokens": tokens,
              "max_sequence_length": max(len(encoded[i]["input_ids"]) for i in indices),
              "sample_count": 8, "good": 5, "bad": 3, **memory_status()}
    if cuda:
        result.update({"peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                       "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved()})
    return result


def worst_case_batches(encoded):
    good = sorted((i for i, r in enumerate(encoded) if r["route_label"] == 0),
                  key=lambda i: (-len(encoded[i]["input_ids"]), encoded[i]["audit_id"]))
    bad = sorted((i for i, r in enumerate(encoded) if r["route_label"] == 1),
                 key=lambda i: (-len(encoded[i]["input_ids"]), encoded[i]["audit_id"]))
    if len(good) < 10 or len(bad) < 6:
        raise ValueError("Two distinct worst-case balanced batches require 10 Good and 6 Bad rows")
    return [good[:5] + bad[:3], good[5:10] + bad[3:6]]


def preflight(model, tokenizer, encoded, core, marker, cfg, identity, receipts, runtime, order):
    import torch
    # No scheduler: both disposable updates use the real nonzero peak LR.  A
    # warmup step with LR=0 must not masquerade as the requested update test.
    offload = make_optimizer(model, cfg)
    measurements = []
    started = time.perf_counter()
    try:
        selected = [("first_sampler_batch_cold_adam", order[0]),
                    ("longest_5_good_3_bad_warm_adam", worst_case_batches(encoded)[0])]
        for number, (kind, indices) in enumerate(selected, start=1):
            result = global_batch(model, offload, encoded, indices, core, marker, cfg)
            result["measurement_kind"] = kind
            result["audit_ids"] = [encoded[i]["audit_id"] for i in indices]
            measurements.append(result)
            emit("disposable_gpu_preflight_batch", batch=number, **result)
        offload.release_gradients()
    finally:
        offload.close()
    slowest = max(r["seconds"] for r in measurements)
    result = {"schema": "webshop-full-parameter-gpu-preflight-v1", "passed": True,
              "identity": identity, "identity_sha256": digest(identity), "runtime": runtime,
              "disposable": True, "formal_training_started": False, "model_weights_saved": False,
              "full_parameter_fp32": True, "bf16_autocast": True,
              "nonreentrant_gradient_checkpointing": True, "global_batch": 8,
              "optimizer_backend": cfg["optimizer_backend"],
              "batches": measurements, "max_measured_batch_seconds": slowest,
              "elapsed_seconds": time.perf_counter() - started,
              "1500_times_max_measured_batch_seconds": 1500 * slowest,
              "typical_sampler_batch_seconds": measurements[0]["seconds"],
              "worst_length_batch_seconds": measurements[1]["seconds"],
              "timing_scope": "First actual sampler batch (cold Adam state initialization) plus longest 5:3 batch (warm Adam), including forward/backward and configured native AdamW update. Not an end-to-end ETA; excludes three evaluations and two checkpoints.",
              "training_rng_preserved": True, "paid_api_calls": 0}
    atomic_json(receipts / "gpu_preflight.json", result)
    emit("gpu_preflight_complete", receipt=str(receipts / "gpu_preflight.json"),
         max_measured_batch_seconds=slowest, **memory_status())
    return result


def assert_training_gate(cfg, receipts, identity):
    if os.environ.get("ALLOW_WEBSHOP_ROUTE_SFT") != "1" or cfg.get("training_authorized") is not True:
        raise PermissionError("Formal training requires config training_authorized=true AND ALLOW_WEBSHOP_ROUTE_SFT=1")
    numerical_name = ("optimizer_numerics.json" if cfg["optimizer_backend"] == "native_torch_adamw_cuda_fp32"
                      else "offload_numerics.json")
    for name in ("tokenizer_preflight.json", numerical_name, "gpu_preflight.json"):
        receipt = read_json(Path(receipts) / name)
        if receipt.get("passed") is not True or receipt.get("identity_sha256") != digest(identity):
            raise ValueError("Missing/stale/failed preflight receipt: " + name)
    gpu = read_json(Path(receipts) / "gpu_preflight.json")
    if gpu.get("disposable") is not True or len(gpu.get("batches", [])) != 2:
        raise ValueError("Two disposable full-model GPU batches have not passed")


def checkpoint_files(directory):
    return {str(p.relative_to(directory)): {"sha256": sha(p), "bytes": p.stat().st_size}
            for p in sorted(Path(directory).rglob("*")) if p.is_file() and p.name != "manifest.json"}


def save_checkpoint(model, tokenizer, offload, scheduler, order, cfg, identity, completed, output, root, *, emergency=False):
    import torch
    eligible = completed in (750, 1500) or (emergency and 0 < completed < 1500)
    if not eligible or offload.active or offload.updates != completed:
        raise ValueError("Full immutable checkpoints are allowed only at complete steps 750 and 1500")
    target = Path(output) / f"checkpoint_{completed:06d}"
    if target.exists():
        raise FileExistsError("Immutable checkpoint already exists; refusing overwrite")
    pending = target.with_name(target.name + ".incomplete-" + uuid.uuid4().hex)
    pending.mkdir(parents=True, exist_ok=False)
    state = rng_state()
    with preserve_rng():
        model.save_pretrained(pending / "model", safe_serialization=True, max_shard_size="2GB")
        tokenizer.save_pretrained(pending / "model")
        atomic_json(pending / "config.json", identity.get("config", cfg))
        atomic_json(pending / "identity.json", identity)
        atomic_json(pending / "batch_order.json", order)
        for name, checksum in identity["code"].items():
            source = within(root, name)
            if sha(source) != checksum:
                raise ValueError("Source changed during checkpoint creation")
            destination = pending / "code_snapshot" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        from deadline_policy import policy
        torch.save({"schema": "webshop-exact-resume-v1", "completed_updates": completed,
                    "next_batch": completed, "optimizer": offload.optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "rng": state, "offload_updates": offload.updates,
                    "parameter_names": [name for name, _ in offload.named],
                    "identity_sha256": digest(identity), "order_sha256": digest(order),
                    "runtime_deadline_policy": policy()},
                   pending / "training_state.pt")
        manifest = {"schema": "webshop-immutable-full-checkpoint-v1", "complete": True,
                    "checkpoint_reason": "shutdown_deadline" if emergency else "scheduled",
                    "completed_updates": completed, "next_batch": completed,
                    "next_batch_audit_ids": [identity["train_audit_ids"][i] for i in order[completed]]
                        if completed < len(order) else [],
                    "identity_sha256": digest(identity), "files": checkpoint_files(pending)}
        # Publication marker is the last file; the final directory appears in
        # one rename. Failed/incomplete directories are retained for inspection.
        atomic_json(pending / "manifest.json", manifest)
        os.rename(pending, target)
    return target


def verify_checkpoint(path, identity, order):
    path = Path(path)
    if ".incomplete-" in path.name:
        raise ValueError("Cannot resume an unpublished checkpoint")
    manifest = read_json(path / "manifest.json")
    completed = manifest.get("completed_updates")
    eligible = completed in (750, 1500) or (manifest.get("checkpoint_reason") == "shutdown_deadline"
                                           and isinstance(completed,int) and 0 < completed < 1500)
    if manifest.get("complete") is not True or not eligible \
            or manifest.get("identity_sha256") != digest(identity):
        raise ValueError("Checkpoint identity/step is not this exact experiment")
    if read_json(path / "identity.json") != identity or read_json(path / "batch_order.json") != order:
        raise ValueError("Checkpoint identity/order differs")
    actual = checkpoint_files(path)
    if actual != manifest["files"]:
        raise ValueError("Incomplete or changed checkpoint contents")
    return manifest


def restore_checkpoint(path, offload, scheduler, identity, order):
    import torch
    manifest = verify_checkpoint(path, identity, order)
    state = torch.load(Path(path) / "training_state.pt", map_location="cpu", weights_only=False)
    completed = manifest["completed_updates"]
    if state["identity_sha256"] != digest(identity) or state["order_sha256"] != digest(order) \
            or state["next_batch"] != completed or state["completed_updates"] != completed \
            or state["offload_updates"] != completed \
            or state["parameter_names"] != [name for name, _ in offload.named]:
        raise ValueError("Optimizer/RNG/sampler checkpoint state differs")
    offload.optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    offload.updates = completed
    restore_rng(state["rng"])
    return completed


def run_evaluation(model, tokenizer, rows, valenc, marker, cfg, step, output):
    from evaluation import run_suite
    from deadline_policy import evaluation_cutoff
    evaluation_cfg = dict(cfg, _evaluation_deadline_epoch=evaluation_cutoff(step, cfg["total_updates"]))
    cutoff = evaluation_cfg['_evaluation_deadline_epoch']
    if cutoff is not None and time.time() >= cutoff:
        atomic_json(Path(output) / "evaluations" / f"step_{step:06d}_deferred.json",
                    {"step":step,"status":"deferred_shutdown_budget","completed":False})
        emit("evaluation_deferred_for_checkpoint_budget",step=step)
        return 0.
    was_training = model.training
    started = time.perf_counter()
    evaluation_output = Path(output) / "evaluations" / f"step_{step:06d}"
    if (evaluation_output / "suite_deferred.json").is_file():
        evaluation_output.rename(evaluation_output.with_name(evaluation_output.name + ".deferred-" + uuid.uuid4().hex))
    try:
        with preserve_rng():
            result = run_suite(model, tokenizer, rows, valenc, marker, evaluation_cfg, step,
                               evaluation_output)
    finally:
        model.train(was_training)
    elapsed = time.perf_counter() - started
    event = "evaluation_complete" if result.get("status") == "complete" else "evaluation_deferred_for_shutdown"
    emit(event, step=step, seconds=elapsed,
         summary=result if isinstance(result, dict) else {"completed": True})
    return elapsed


def train(model, tokenizer, core, marker, trainenc, val, valenc, order, cfg, identity,
          output, root, resume=None):
    # Verify the entire evaluation module can be imported before step 1.
    from evaluation import run_suite  # noqa: F401
    offload = make_optimizer(model, cfg)
    scheduler = make_scheduler(offload.optimizer, cfg["warmup_steps"], cfg["total_updates"])
    start = restore_checkpoint(resume, offload, scheduler, identity, order) if resume else 0
    Path(output).mkdir(parents=True, exist_ok=True)
    # A resumed run can replay steps after its checkpoint. Preserve the old
    # trajectory, never append ambiguous duplicate step numbers to its history.
    history = Path(output) / (f"training_resume_from_{start:06d}_{uuid.uuid4().hex}.jsonl"
                              if resume else "training.jsonl")
    if not resume and history.exists():
        raise FileExistsError("Run already has training history; supply an exact checkpoint to resume")
    started = time.perf_counter()
    from deadline_policy import stop_epoch, policy
    completed = start
    atomic_json(Path(output) / "runtime_deadline_policy.json", policy())
    try:
        if resume:
            emit("exact_checkpoint_resumed", completed_updates=start, history=str(history))
        if start == 0 and 0 in cfg["evaluation_steps"]:
            run_evaluation(model, tokenizer, val, valenc, marker, cfg, 0, output)
        elif start in cfg['evaluation_steps'] and not (Path(output) / "evaluations" / f"step_{start:06d}" / "suite_complete.json").is_file():
            # A checkpoint is published before evaluation. A crash in evaluation
            # must not silently skip this required boundary on resume.
            run_evaluation(model, tokenizer, val, valenc, marker, cfg, start, output)
        for index in range(start, len(order)):
            deadline = stop_epoch()
            if deadline is not None and time.time() >= deadline:
                offload.release_gradients()
                checkpoint = None
                if completed > 0:
                    existing = Path(output) / f"checkpoint_{completed:06d}"
                    if existing.exists():
                        verify_checkpoint(existing,identity,order)
                        checkpoint = existing
                    else:
                        checkpoint = save_checkpoint(model,tokenizer,offload,scheduler,order,cfg,
                            identity,completed,output,root,emergency=True)
                atomic_json(Path(output) / "training_stopped_before_shutdown.json",
                    {"completed_updates":completed,"training_complete":False,
                     "checkpoint":str(checkpoint) if checkpoint else None,"runtime_deadline_policy":policy()})
                emit("shutdown_checkpoint_secured",completed_updates=completed,
                     checkpoint=str(checkpoint) if checkpoint else None)
                return
            lr = offload.optimizer.param_groups[0]["lr"]
            result = global_batch(model, offload, trainenc, order[index], core, marker, cfg)
            scheduler.step()
            completed = index + 1
            payload = {"step": completed, "lr_used": lr, "next_lr": offload.optimizer.param_groups[0]["lr"], **result}
            with history.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            emit("training_update", **payload)
            if completed in cfg["checkpoint_steps"]:
                offload.release_gradients()
                checkpoint_begin = time.perf_counter()
                path = save_checkpoint(model, tokenizer, offload, scheduler, order, cfg,
                                       identity, completed, output, root)
                emit("checkpoint_published", step=completed, path=str(path),
                     seconds=time.perf_counter() - checkpoint_begin)
                run_evaluation(model, tokenizer, val, valenc, marker, cfg, completed, output)
        atomic_json(Path(output) / "training_complete.json", {"complete": True,
            "completed_updates": 1500, "identity_sha256": digest(identity),
            "elapsed_this_invocation_seconds": time.perf_counter() - started,
            "checkpoint_steps": cfg["checkpoint_steps"], "evaluation_steps": cfg["evaluation_steps"],
            "runtime_deadline_policy":policy(),
            "deferred_evaluation_steps":[s for s in cfg['evaluation_steps'] if not
                (Path(output)/'evaluations'/f'step_{s:06d}'/'suite_complete.json').is_file()]})
        emit("training_complete", completed_updates=1500)
    finally:
        offload.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--gpu-preflight-only", action="store_true")
    mode.add_argument("--train", action="store_true")
    parser.add_argument("--resume-checkpoint")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    if str(root) != EXPECTED_ROOT:
        raise ValueError("This launcher is scoped to the explicitly approved remote project root")
    cfg = load_config(args.config)
    if args.resume_checkpoint and not args.train:
        raise ValueError("Resume is a formal-training operation, not a disposable preflight")
    if args.train and (cfg.get("training_authorized") is not True or os.environ.get("ALLOW_WEBSHOP_ROUTE_SFT") != "1"):
        raise PermissionError("Formal training is not armed; no model was loaded")
    runtime = runtime_check(cfg)
    emit("runtime_verified", **runtime)
    core, reference_identity = verify_reference(root, cfg)
    parent_identity = verify_parent(root, cfg)
    rows, val, order, data_identity = validate_dataset(root, cfg, core)
    identity = {"schema": "webshop-sft-exact-identity-v1", "config": cfg,
                "config_sha256": digest(cfg), "reference": reference_identity,
                "parent": parent_identity, "data": data_identity, "code": code_identity(root, cfg),
                "train_audit_ids": [r["audit_id"] for r in rows]}
    receipts = within(root, cfg["receipts_path"])
    output = within(root, cfg["output_path"])
    # Runtime-only additions are not part of the experiment's identity.
    eval_cfg = dict(cfg, root=str(root), device="cuda:0")
    if args.train:
        assert_training_gate(cfg, receipts, identity)
    from transformers import AutoTokenizer
    resume = within(root, args.resume_checkpoint) if args.resume_checkpoint else None
    if resume:
        verify_checkpoint(resume, identity, order)
    model_path = resume / "model" if resume else within(root, cfg["model_path"])
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    token_state = core.register_token(tokenizer)
    trainenc = encode_all(rows, tokenizer, core, cfg["max_length"])
    valenc = encode_all(val, tokenizer, core, cfg["max_length"])
    tokenizer_receipt = {"passed": True, "identity_sha256": digest(identity),
                         "train_rows": len(trainenc), "validation_rows": len(valenc),
                         "joint_tokenization_matches_all_rows": True, "marker": token_state,
                         "max_train_length": max(len(r["input_ids"]) for r in trainenc),
                         "max_validation_length": max(len(r["input_ids"]) for r in valenc)}
    if args.gpu_preflight_only:
        atomic_json(receipts / "tokenizer_preflight.json", tokenizer_receipt)
    emit("all_rows_tokenizer_preflight", **tokenizer_receipt)
    seed_all(cfg["seed"])
    if args.gpu_preflight_only:
        # Both the tiny parity check and the full-model disposable updates leave
        # this process's Python/CPU/CUDA RNG states as they found them.
        with preserve_rng():
            resident = cfg["optimizer_backend"] == "native_torch_adamw_cuda_fp32"
            numerical = verify_resident_numerics("cuda:0") if resident else verify_offload_numerics("cuda:0")
            numerical_name = "optimizer_numerics.json" if resident else "offload_numerics.json"
            atomic_json(receipts / numerical_name, {**numerical,
                        "identity_sha256": digest(identity)})
            emit("synthetic_optimizer_numerics", **numerical)
            model, marker = load_model(model_path, tokenizer, core)
            try:
                preflight(model, tokenizer, trainenc, core, marker["token_id"], cfg,
                          identity, receipts, runtime, order)
            finally:
                del model
                gc.collect()
                import torch
                torch.cuda.empty_cache()
        return 0
    model, marker = load_model(model_path, tokenizer, core)
    train(model, tokenizer, core, marker["token_id"], trainenc, val, valenc, order,
          eval_cfg, identity, output, root, resume=resume)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        emit("failed", error_type=type(exc).__name__, message=str(exc), formal_training_completed=False)
        raise
