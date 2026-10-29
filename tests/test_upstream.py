from pathlib import Path
import pytest
from rethink_webshop.common import UPSTREAM,load,ROOT


def test_minimal_patch_and_native_losses():
    lock=load('upstream.json')
    assert set(lock['patched_files'])=={'agent_system/multi_turn_rollout/rollout_loop.py',
          'agent_system/environments/env_package/webshop/envs.py','verl/trainer/ppo/skillsd_ray_trainer.py'}
    assert (ROOT/'patches/sdar.patch').stat().st_size<10000
    if not UPSTREAM.exists():
        pytest.skip('Prepare public SDAR to run the source-integrity check')
    from rethink_webshop.setup import verify_framework
    verify_framework()


def test_source_contains_no_student_hint_or_loss_override():
    patch=(ROOT/'patches/sdar.patch').read_text()
    assert 'skip_special_tokens=False' in patch
    assert 'score_teacher' in patch
    assert 'sdar_utils.py' not in patch and 'dp_actor.py' not in patch
    assert 'compute_advantage' not in patch and 'use_kl_loss' not in patch
