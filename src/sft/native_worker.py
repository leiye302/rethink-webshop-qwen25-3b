"""CPU-only pinned NativeWorld JSONL worker, launched by evaluation.py.

stdout is reserved for JSONL even when Java or native dependencies write to fd 1.
All model generation stays in the parent GPU process.  This worker never imports
torch, Ray, verl, an Actor server, or a teacher/analyzer API.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback


UPSTREAM_COMMIT = "d511f043afe9199b8576fc99ea52e12d84841f32"


class NativeSessionWorker:
    def __init__(self, root):
        from webshop_data.common import config, digest
        from webshop_data.native import NativeWorld, source_identity
        settings = config()
        expected = {"upstream_commit": UPSTREAM_COMMIT, "environment_seed": 0,
                    "history_length": 2, "use_small": True, "human_goals": False,
                    "full_catalog_required": False}
        if any(settings.get(key) != value for key, value in expected.items()):
            raise ValueError("Native configuration differs from the agreed pinned synthetic-goal protocol")
        self.root = Path(root).resolve()
        self.world = NativeWorld(self.root / ".deps/sdar", self.root / ".deps/sdar/agent_system/environments/env_package/webshop/webshop/data")
        self.digest = digest
        self.native_sources = source_identity(self.root / ".deps/sdar")
        if len(self.world.goals) < 500 or self.world.identity["products"] != 1000:
            raise ValueError("Expected 1000 products and at least 500 synthetic goals")
        self.sessions = {}
        self.seen = set()

    @property
    def ready(self):
        return {"event": "ready", "identity": self.world.identity,
                "native_sources": self.native_sources, "cpu_only": True,
                "actor_receives_hidden_goal": False}

    def reset(self, items):
        result = []
        identifiers = [item["session_id"] for item in items]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Duplicate reset session ID")
        if len(self.sessions) + len(items) > 16:
            raise ValueError("At most 16 native episodes may be live")
        # Validate the entire request before performing any environment reset.
        for item in items:
            session, task = item["session_id"], item["task_id"]
            if not isinstance(session, str) or not session or session in self.seen:
                raise ValueError("Native sessions must have unique, nonempty IDs")
            if not isinstance(task, int) or isinstance(task, bool) or not 0 <= task < 500:
                raise ValueError("Native eval goals must come from reserved IDs 0..499")
        for item in items:
            session, task = item["session_id"], item["task_id"]
            env, view, obs = self.world.reset(task, session)
            try:
                state = self.world.visible(env, view, obs)
            except BaseException:
                env.close()
                raise
            self.sessions[session] = {"env": env, "view": view, "state": state, "done": False, "steps": 0}
            self.seen.add(session)
            # Hashes bind identity without disclosing private goal attributes or
            # future answers.  Only state.prompt reaches the model in the parent.
            goal = self.world.goals[task]
            result.append({"session_id": session, "task_id": task, "state": state,
                           "task_sha256": self.digest(goal),
                           "instruction_sha256": self.digest(goal["instruction_text"])})
        return {"items": result}

    def step(self, items):
        identifiers = [item["session_id"] for item in items]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("A native step batch may act on each session only once")
        for item in items:
            session = self.sessions[item["session_id"]]
            if session["done"] or session["steps"] >= 15:
                raise ValueError("Attempt to step a terminal or 15-step episode")
            if not isinstance(item["response"], str):
                raise ValueError("Only literal generated response text may be projected")
            if item.get("expected_state_sha256") != session["state"]["state_sha256"]:
                raise ValueError("Native pre-action state hash changed")
        result = []
        for item in items:
            session_id, response = item["session_id"], item["response"]
            session = self.sessions[session_id]
            before = session["state"]
            # The unchanged pinned projector performs its own extraction,
            # lowercase/strip, and last-20-character malformed fallback.  Do not
            # replace malformed actions or add protocol/availability gates.
            transition = self.world.step(session["env"], session["view"],
                                         before["formatted_observation"], response)
            session["steps"] += 1
            session["done"] = transition["done"]
            lower = response.lower()
            fallback = lower.find("<action>") == -1 or lower.find("</action>") == -1
            next_state = None
            if not transition["done"]:
                next_state = self.world.visible(session["env"], session["view"],
                                                transition["next_observation"])
                session["state"] = next_state
            result.append({"session_id": session_id, "result": transition,
                           "projection_used_fallback": fallback, "next_state": next_state})
        return {"items": result}

    def close_sessions(self, identifiers):
        for session_id in identifiers:
            session = self.sessions.pop(session_id, None)
            if session is not None:
                session["env"].close()
        return {"closed_session_ids": list(identifiers)}

    def close(self):
        errors = []
        for session_id in list(self.sessions):
            try:
                self.close_sessions([session_id])
            except Exception as error:
                errors.append(str(error))
        try:
            self.world.base.close()
        except Exception as error:
            errors.append(str(error))
        if errors:
            raise RuntimeError("Native environment close failed: " + "; ".join(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    # Duplicate the original pipe first. Redirect fd 1 as well as Python stdout
    # BEFORE loading Java/native code, so native print calls cannot corrupt IPC.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1, encoding="utf-8")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    worker = None

    def send(payload):
        protocol.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
        protocol.flush()

    try:
        worker = NativeSessionWorker(args.root)
        send({"ok": True, **worker.ready})
        for line in sys.stdin:
            request = {}
            try:
                request = json.loads(line)
                operation = request["op"]
                if operation == "reset":
                    payload = worker.reset(request["items"])
                elif operation == "step":
                    payload = worker.step(request["items"])
                elif operation == "close" and "session_ids" in request:
                    payload = worker.close_sessions(request["session_ids"])
                elif operation == "close":
                    worker.close()
                    worker = None
                    send({"ok": True, "request_id": request.get("request_id"), "event": "closed"})
                    return 0
                else:
                    raise ValueError("Unknown native worker operation: " + str(operation))
                send({"ok": True, "request_id": request.get("request_id"), **payload})
            except Exception as error:
                traceback.print_exc(file=sys.stderr)
                send({"ok": False, "request_id": request.get("request_id"),
                      "error_type": type(error).__name__, "error": str(error)})
                # No retries or partial-batch recovery: parent receives failure
                # and still has a working close command in its finally block.
    except BaseException as error:
        traceback.print_exc(file=sys.stderr)
        send({"ok": False, "event": "initialization_error", "error_type": type(error).__name__,
              "error": str(error)})
        return 1
    finally:
        if worker is not None:
            try:
                worker.close()
            except Exception:
                traceback.print_exc(file=sys.stderr)
        protocol.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
