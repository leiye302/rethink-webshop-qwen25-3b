"""Same reserved goals, identities and seed for baseline/method and both sizes."""
from .common import load


def seed_for_worker(seed, index, group_n, is_train):
    return seed+index//group_n if is_train else 0


def goal_indices(rng, goal_idxs, env_num, is_train):
    if is_train:
        return rng.choice(goal_idxs,size=env_num,replace=False)
    import numpy as np
    tasks=load('eval128.json')['task_ids']
    if env_num!=len(tasks) or len(set(tasks))!=128 or any(not 0<=x<500 for x in tasks):
        raise ValueError('Evaluation must use the identical reserved 128 native goals')
    return np.array(tasks,dtype=np.int64)
