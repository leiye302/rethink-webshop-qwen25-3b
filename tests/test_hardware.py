import math
import pytest
from rethink_webshop.hardware import validate_properties,validate_visible_devices
from rethink_webshop.common import load,UPSTREAM


def test_hardware_contract_without_loading_cuda():
    validate_properties([('NVIDIA A100-SXM4-80GB',80*2**30)]*4)
    validate_properties([('NVIDIA A100 80GB PCIe',80*2**30)]*4)
    for bad in [[('NVIDIA A100',40*2**30)]*4,[('NVIDIA A800',80*2**30)]*4,
                [('NVIDIA A100',80*2**30)]*2]:
        with pytest.raises(RuntimeError): validate_properties(bad)
    assert validate_visible_devices('0,1,2,3')==['0','1','2','3']
    for bad in ['0,1','0,1,1,2','0,1,2,3,4','-1']:
        with pytest.raises(ValueError): validate_visible_devices(bad)


def test_native_batch_padding_rule_matches_the_two_gpu_recipe():
    cfg=load('opsd.json')
    old=math.lcm(16*2,16*2,8*2)
    new=math.lcm(cfg['actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu']*4,
                 cfg['actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu']*4,
                 cfg['actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu']*4)
    assert old==new==32
    for size in range(1,1600):
        assert (-size)%old==(-size)%new


def test_resolved_method_and_baseline_share_the_four_gpu_recipe(tmp_path):
    if not UPSTREAM.exists(): pytest.skip('Prepare pinned SDAR first')
    pytest.importorskip('omegaconf')
    from omegaconf import OmegaConf
    from rethink_webshop.run import resolved
    a=OmegaConf.to_container(resolved(tmp_path/'actor',tmp_path/'run',False),resolve=True)
    b=OmegaConf.to_container(resolved(tmp_path/'actor',tmp_path/'run',True),resolve=True)
    assert a['trainer']['n_gpus_per_node']==b['trainer']['n_gpus_per_node']==4
    assert a['actor_rollout_ref']['rollout']['tensor_model_parallel_size']==2
    a['algorithm']['local_hints']['enabled']=b['algorithm']['local_hints']['enabled']
    a['trainer']['experiment_name']=b['trainer']['experiment_name']
    assert a==b
