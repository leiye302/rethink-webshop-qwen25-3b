from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

from .common import ROOT, config, digest, file_sha


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native_view(sdar, history_length=2):
    """Run the exact pinned prompt/memory methods without importing Ray/verl."""
    sdar = Path(sdar)
    prompts = load_module(sdar / "agent_system/environments/prompts/webshop.py", "webshop_native_prompt")
    ns = {"List": List, "Dict": Dict, "Any": Any, "Tuple": Tuple,
          "BaseMemory": object, "EnvironmentManagerBase": object,
          "WEBSHOP_TEMPLATE": prompts.WEBSHOP_TEMPLATE,
          "WEBSHOP_TEMPLATE_NO_HIS": prompts.WEBSHOP_TEMPLATE_NO_HIS}
    tree = ast.parse((sdar / "agent_system/memory/memory.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SimpleMemory")
    exec(compile(ast.Module(body=[node], type_ignores=[]), "pinned_sdar_memory", "exec"), ns)
    tree = ast.parse((sdar / "agent_system/environments/env_manager.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WebshopEnvironmentManager")
    methods = {"extract_task", "format_obs", "format_avail_actions", "build_text_obs"}
    node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    exec(compile(ast.Module(body=[node], type_ignores=[]), "pinned_sdar_webshop_view", "exec"), ns)
    view = ns["WebshopEnvironmentManager"]()
    view.memory = ns["SimpleMemory"]()
    view.memory.reset(1)
    view.config = SimpleNamespace(env=SimpleNamespace(history_length=history_length))
    return view


class NativeWorld:
    def __init__(self, sdar, assets):
        import sys
        self.sdar, self.assets = Path(sdar).resolve(), Path(assets).resolve()
        import json
        locked = json.loads((ROOT / "configs/native_source.lock.json").read_text())
        if source_identity(self.sdar) != locked["files"]:
            raise ValueError("Native SDAR prompt/environment/score source does not match the pinned SHA")
        self.webshop = self.sdar / "agent_system/environments/env_package/webshop/webshop"
        sys.path.insert(0, str(self.webshop))
        from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
        self.env_class = WebAgentTextEnv
        self.projection = load_module(self.sdar / "agent_system/environments/env_package/webshop/projection.py",
                                      "pinned_webshop_projection").webshop_projection
        c = config()
        catalog = "items_shuffle_1000.json" if c["use_small"] else "items_shuffle.json"
        attributes = "items_ins_v2_1000.json" if c["use_small"] else "items_ins_v2.json"
        self.kwargs = {"observation_mode": "text", "seed": c["environment_seed"],
                       "file_path": str(self.assets / catalog),
                       "attr_path": str(self.assets / attributes),
                       "num_products": None, "human_goals": c["human_goals"], "show_attrs": False}
        # The native HUMAN_ATTR_PATH is data/items_human_ins.json, installed as
        # a link to the verified assets by prepare_resources, never a hidden goal
        # or a substitute target passed to the Actor.
        self.base = self.env_class(**self.kwargs)
        if c["full_catalog_required"] and len(self.base.server.all_products) <= 1_000_000:
            raise ValueError("Refusing to label a 1k preview as the full WebShop catalog")
        self.goals = self.base.server.goals
        self.identity = {"products": len(self.base.server.all_products), "goals": len(self.goals),
                         "goal_order_sha256": digest(self.goals), "human_goals": c["human_goals"],
                         "use_small": c["use_small"], "environment_profile": c["environment_profile"],
                         "environment_seed": c["environment_seed"], "upstream_commit": c["upstream_commit"]}

    def reset(self, task_id, episode_id):
        env = self.env_class(**self.kwargs, server=self.base.server, session_prefix=episode_id + "_")
        obs, _ = env.reset(session=int(task_id))
        view = native_view(self.sdar, config()["history_length"])
        view.tasks = view.extract_task([obs])
        return env, view, obs

    def visible(self, env, view, obs):
        formatted = view.format_obs([obs])[0]
        available = env.get_available_actions()
        prompt = view.build_text_obs([formatted], [{"available_actions": available}],
                                     init=not view.memory[0])[0]
        commands = view.format_avail_actions(available)
        # No private goal, sampled action, future observation, or label enters H.
        state = {"prompt": prompt, "observation": obs, "available_actions": available}
        return {**state, "formatted_observation": formatted, "admissible_commands": commands,
                "state_sha256": digest(state)}

    def step(self, env, view, formatted, response):
        actions, valids = self.projection([response])
        action = actions[0]
        # Never execute environment-internal attribute-query shortcuts.
        if any(x in action for x in ("<a>", "<q>", "<c>", "<r>")):
            raise ValueError("Hidden search operator is not an ordinary online action")
        obs, raw, done, _ = env.step(action)
        view.memory.store({"text_obs": [formatted], "action": [action]})
        return {"next_observation": obs, "raw_task_score": float(raw), "done": bool(done),
                "won": bool(done and raw == 1.0), "training_reward": 10.0 if done and raw == 1.0 else 0.0,
                "environment_action": action, "native_format_valid": bool(valids[0])}


def source_identity(sdar):
    root = Path(sdar)
    names = ["agent_system/environments/prompts/webshop.py", "agent_system/memory/memory.py",
             "agent_system/environments/env_manager.py", "agent_system/environments/env_package/webshop/projection.py",
             "agent_system/environments/env_package/webshop/envs.py", "verl/trainer/ppo/sdar_utils.py",
             "examples/grpo_opsd_trainer/run_webshop_3b.sh", "verl/trainer/config/ppo_trainer.yaml"]
    return {name: file_sha(root / name) for name in names}
