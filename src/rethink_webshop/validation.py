"""Compatibility helpers for the shared140, content-filtered environment."""
from .fixed_validation import manifest


def seed_for_worker(seed, index, group_n, is_train):
    return seed + index // group_n


def goal_indices(rng, goal_idxs, env_num, is_train):
    if is_train:
        return rng.choice(goal_idxs, size=env_num, replace=False)
    import numpy as np
    fixed = manifest()
    if env_num != 140 or list(goal_idxs) != list(range(140)):
        raise ValueError('Evaluation requires the same shared140 content-filtered goals')
    return np.array([r['goal_index'] for r in fixed['selection']], dtype=np.int64)
