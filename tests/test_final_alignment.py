import json
from rethink_webshop.common import ROOT,sha

def test_final_alfworld_protocol_sources_are_exactly_pinned():
    lock=json.loads((ROOT/'configs/hint_protocol.json').read_text())
    for name,expected in lock['source_files'].items():
        assert sha(ROOT/'src/rethink_webshop'/name)==expected
    assert lock['model']=='glm-5.3-flash' and not lock['analyzer_is_router']

def test_hint_source_is_identical_for_both_model_sizes():
    # This checkout has no dependency on the other checkout: the contract is
    # checked against its own pinned source files above.
    lock=json.loads((ROOT/'configs/hint_protocol.json').read_text())
    assert lock['teacher']=='no_grad_pre_update_current_actor'
    assert not lock['student_sees_hint']
