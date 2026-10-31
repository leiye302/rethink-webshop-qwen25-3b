import ast
import copy
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from rethink_webshop.common import ROOT, UPSTREAM

def test_actual_native_vector_reset_frozen_and_training_random_on_filtered_pool(monkeypatch):
    from rethink_webshop.fixed_validation import reset_fixed_validation, manifest
    source = UPSTREAM/'agent_system/environments/env_package/webshop/envs.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WebshopMultiProcessEnv')
    class Worker:
        def __init__(self, seed, kwargs, is_train): self.seed, self.is_train = seed, is_train
        def get_goals(self): return list(range(1000 if self.is_train else 140))
        def reset(self, index):
            key = 'training' if self.is_train else manifest()['selection'][index]['task_id']
            return f'{self.seed}:{index}', {'task_id': key}
        def close(self): pass
    class Handle:
        def __init__(self, obj): self.obj = obj
        def __getattr__(self, name): return SimpleNamespace(remote=getattr(self.obj, name))
    remote = SimpleNamespace(remote=lambda *a:Handle(Worker(*a)))
    ray = SimpleNamespace(is_initialized=lambda:True, remote=lambda **kw:lambda c:remote,
                          get=lambda x:x, kill=lambda x:None)
    namespace = {'gym':SimpleNamespace(Env=object), 'np':np, 'ray':ray, 'WebshopWorker':Worker}
    monkeypatch.setitem(sys.modules, 'rethink_webshop.webshop_worker', SimpleNamespace(DisjointWebshopWorker=Worker))
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(source),'exec'),namespace)
    Native = namespace['WebshopMultiProcessEnv']
    training = Native(0,16,8,{},is_train=True)
    validation = Native(1000,140,1,{},is_train=False)
    try:
        first_train = training.reset()
        val1, identity1 = reset_fixed_validation(validation, validation.reset, 1000)
        val2, identity2 = reset_fixed_validation(validation, validation.reset, 1000)
        assert val1 == val2 and identity1 == identity2
        assert val1[0] == [f"{r['worker_seed']}:{r['goal_index']}" for r in manifest()['selection']]
        second_train = training.reset()
        control = np.random.RandomState(0)
        first = np.repeat(control.choice(range(1000),16,replace=False),8)
        second = np.repeat(control.choice(range(1000),16,replace=False),8)
        assert first_train[0] == [f'{i//8}:{x}' for i,x in enumerate(first)]
        assert second_train[0] == [f'{i//8}:{x}' for i,x in enumerate(second)]
        assert first_train != second_train
    finally:
        training.close()
        validation.close()


def test_changed_initial_goal_is_rejected_not_silently_compared(tmp_path):
    from rethink_webshop.fixed_validation import record_identity
    identity = {'manifest_sha256':'same', 'initial_observations_sha256':'original'}
    record_identity(tmp_path, identity)
    record_identity(tmp_path, identity)
    with pytest.raises(RuntimeError, match='initial tasks changed'):
        record_identity(tmp_path, dict(identity, initial_observations_sha256='different'))


@pytest.fixture
def synthetic_split(monkeypatch):
    from rethink_webshop import task_split as split
    goals = []
    for category in split.CATEGORIES:
        for index in range(40):
            goals.append({'asin': f'{category}-{index}', 'name': f'{category} item{index}',
                'category': category, 'query': f'query-{index%12}',
                'product_category': f'{category}/subcategory-{index%15}',
                'instruction_text': f'Find {category} item{index}, and price lower than 50.00 dollars',
                'price_upper': 50., 'weight': 1.,
                'attributes': [f'attribute-{i}' for i in range(2+index%4)],
                'goal_options': {f'option-{i}':'value' for i in range(index%4)}})
    value = split.make_manifest(goals)
    monkeypatch.setattr(split, 'manifest', lambda: value)
    split.validation_ids.cache_clear()
    try:
        yield split, goals, value
    finally:
        split.validation_ids.cache_clear()


def test_balanced_selection_independent_of_native_order_and_price_rng(synthetic_split):
    split, goals, value = synthetic_split
    reversed_goals = copy.deepcopy(goals[::-1])
    for goal in reversed_goals:
        goal['price_upper'] = 90.
        goal['instruction_text'] = goal['instruction_text'].replace('50.00', '90.00')
    repeated = split.make_manifest(reversed_goals)
    assert value['validation_task_ids'] == repeated['validation_task_ids']
    assert value['diversity'] == repeated['diversity']
    before = np.random.RandomState(2).get_state()
    random_state = __import__('random').getstate()
    split.make_manifest(goals)
    assert __import__('random').getstate() == random_state


def test_all_worker_orders_exclude_same140_and_preserve_other_entries(synthetic_split):
    split, goals, value = synthetic_split
    val = set(value['validation_task_ids'])
    for seed in (0,1,15,1000,1139):
        order = np.random.RandomState(seed).permutation(len(goals))
        rows = [copy.deepcopy(goals[i]) for i in order]
        for goal in rows:
            goal['instruction_text'] = goal['instruction_text'].replace('50.00', f'{60+seed%30}.00')
        training, validation = split.partition_goals(rows, True), split.partition_goals(rows, False)
        train_ids = {split.task_identity(g) for g in training}
        assert not train_ids & val
        assert train_ids | val == {split.task_identity(g) for g in goals}
        assert [split.task_identity(g) for g in validation] == value['validation_task_ids']
        assert training == [g for g in rows if split.task_identity(g) not in val]


def test_price_and_duplicate_product_alias_cannot_leak_into_training(synthetic_split):
    split, goals, value = synthetic_split
    first = next(g for g in goals if split.task_identity(g) == value['validation_task_ids'][0])
    alias = dict(copy.deepcopy(first), asin='alias-product', price_upper=70.)
    alias['instruction_text'] = alias['instruction_text'].replace('50.00', '70.00')
    assert split.task_identity(alias) == split.task_identity(first)
    with pytest.raises(RuntimeError, match='leakage'):
        split.assert_allowed(alias, True)
    assert split.assert_allowed(alias, False) == split.task_identity(first)
    changed = dict(alias, instruction_text='A genuinely different requested objective')
    assert split.task_identity(changed) != split.task_identity(first)


def test_catalog_drift_rejected_before_policy_interaction(synthetic_split):
    split, goals, _ = synthetic_split
    with pytest.raises(RuntimeError, match='catalog differs'):
        split.partition_goals(goals[:-1], True)
    changed = copy.deepcopy(goals)
    changed[0]['attributes'].append('new-constraint')
    with pytest.raises(RuntimeError, match='catalog differs'):
        split.partition_goals(changed, False)


def test_wrong_actual_task_binding_rejected_before_generation():
    from rethink_webshop.fixed_validation import reset_fixed_validation, manifest
    env = SimpleNamespace(_rng=np.random.RandomState(1000), goal_idxs=range(140),
        env_num=140, group_n=1, is_train=False)
    previous = env._rng
    def wrong():
        return ['observation']*140, [{'task_id':'wrong'}]*140
    with pytest.raises(RuntimeError, match='Actual validation task identities'):
        reset_fixed_validation(env, wrong, 1000)
    assert env._rng is previous


@pytest.mark.parametrize('is_train', [True, False])
def test_worker_filters_constructor_and_automatic_resets(synthetic_split, is_train):
    split, goals, value = synthetic_split
    class Native:
        def __init__(self, seed, kwargs):
            self.env = SimpleNamespace(server=SimpleNamespace(goals=copy.deepcopy(goals), user_sessions={}))
            self.env.unwrapped = self.env
            self.env.session = 'session'
            def reset(session=0):
                self.env.server.user_sessions['session'] = {'goal': self.env.server.goals[session]}
                return 'observation', None
            self.env.reset = reset
            reset()
        def reset(self, index):
            observation, _ = self.env.reset(index)
            return observation, {}
        def step(self, action):
            self.env.reset(len(self.env.server.goals)-1)  # Simulates native terminal auto-reset.
            return 'terminal observation', 10., True, {'won':True}
    source = ROOT/'src/rethink_webshop/webshop_worker.py'
    cls = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef))
    scope = {'NativeWebshopWorker':Native, 'np':np, 'manifest':split.manifest,
        'partition_goals':split.partition_goals, 'assert_allowed':split.assert_allowed,
        'task_identity':split.task_identity}
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(source),'exec'),scope)
    worker = scope['DisjointWebshopWorker'](0, {}, is_train)
    audit = worker.get_split_audit()
    assert audit['opposite_split_overlap'] == 0
    assert audit['goal_entries'] == (len(goals)-140 if is_train else 140)
    _, info = worker.reset(0)
    before = info['task_id']
    _, reward, done, info = worker.step('click[buy now]')
    assert reward == 10. and done and info['task_id'] == before
    assert worker._current_id() == split.task_identity(worker.env.server.goals[-1])
    with pytest.raises(RuntimeError, match='outside this split'):
        worker.reset(len(worker.env.server.goals))


@pytest.mark.parametrize('raw,done,expected', [(1.,True,10.),(.8,True,0.),(1.,False,0.),(0.,False,0.)])
def test_original_webshop_reward_not_changed(raw, done, expected):
    path = UPSTREAM/'agent_system/environments/env_package/webshop/envs.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name == 'WebshopWorker')
    space = {}
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(path),'exec'),space)
    worker = space['WebshopWorker'].__new__(space['WebshopWorker'])
    worker.env = SimpleNamespace(step=lambda a:('observation',raw,done,{}),
        get_available_actions=lambda:{'clickables':['buy now']})
    obs,reward,ended,info = worker.step('click[buy now]')
    assert reward == expected and ended == done and info['task_score'] == raw
    assert info['won'] == (expected == 10.)
