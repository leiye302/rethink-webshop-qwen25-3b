from __future__ import annotations

import hashlib
import contextlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
MARKER = "<RETHINK>"


@contextlib.contextmanager
def exclusive_run(directory, name=".run.lock"):
    """OS-owned lock: an interrupted process releases it, without blind restart."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    handle = (root / name).open("a+")
    handle.seek(0)
    if os.name == "nt":
        import msvcrt
        if os.fstat(handle.fileno()).st_size == 0:
            handle.write(" ")
            handle.flush()
        handle.seek(0)
        lock = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        unlock = lambda: (handle.seek(0), msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1))
    else:
        import fcntl
        lock = lambda: fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        unlock = lambda: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    acquired = False
    try:
        try:
            lock()
            acquired = True
        except OSError:
            raise RuntimeError("This output already has a live collector/annotation process") from None
        yield
    finally:
        if acquired:
            unlock()
        handle.close()


def config():
    return json.loads((ROOT / "configs/native.json").read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".pending")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".pending")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def parse_response(text):
    # No protocol repair, injected reasoning, action duplication, or prefilled assistant.
    body = text.removesuffix("<|im_end|>").removesuffix("<|endoftext|>").strip()
    pattern = r"(?P<prefix><think>(?P<reasoning>[\s\S]*?\S)\s*</think>\s*)(?P<marker><RETHINK>)?\s*<action>(?P<action>[^\r\n]+?)</action>"
    match = re.fullmatch(pattern, body)
    counts_ok = all(body.count(tag) == 1 for tag in ("<think>", "</think>", "<action>", "</action>"))
    if not match or not counts_ok or body.count(MARKER) != int(match.group("marker") is not None):
        return {"strict_protocol": False, "marker_present": MARKER in body}
    return {"strict_protocol": True, "marker_present": bool(match.group("marker")),
            "reasoning": match.group("reasoning"), "source_prefix": match.group("prefix"),
            "sampled_action": match.group("action")}


def action_admissible(action, available):
    if not isinstance(action, str) or any(c in action for c in "\r\n"):
        return False
    if any(tag in action for tag in ("<think>", "</think>", "<action>", "</action>", MARKER, "<|")):
        return False
    match = re.fullmatch(r"(search|click)\[([^\r\n]+)\]", action.strip(), re.I)
    if not match:
        return False
    kind, arg = match[1].lower(), match[2].strip().lower()
    if kind == "search":
        # The native engine admits arbitrary queries, but hidden internal query
        # operators are never exposed as an ordinary Actor action.
        return bool(available["has_search_bar"] and arg and not re.search(r"<[aqcr]>", arg))
    return arg != "search" and arg in {str(x).lower() for x in available["clickables"]}
