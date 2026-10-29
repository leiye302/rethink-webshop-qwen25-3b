"""Frozen greedy 200-prompt/128-episode evaluation using the SFT evaluation engine."""
import argparse
from pathlib import Path
import sys
from .common import ROOT,confined,load


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,help='Project-relative HF model directory')
    p.add_argument('--output',required=True,help='New project-relative evaluation directory')
    a=p.parse_args()
    sys.path.insert(0,str(ROOT/'src/sft'))
    sys.path.insert(0,str(ROOT/'src/sft_reference'))
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from sft_reference.route_bce import core
    from evaluation import run_suite
    from .setup import verify_framework
    verify_framework()
    target=confined(a.model);output=confined(a.output)
    if output.exists():
        raise FileExistsError('Use a new immutable evaluation output directory')
    tok=AutoTokenizer.from_pretrained(target,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(target,torch_dtype=torch.float32,
                  attn_implementation='sdpa',local_files_only=True).to('cuda:0')
    marker=core.register_token(tok,model=model)['token_id']
    dataset=ROOT/'datasets/frozen_sft'
    rows=core.load_rows(dataset/'val.jsonl')
    encoded=[core.encode_row(row,tok,8192) for row in rows]
    cfg=dict(load('sft.json'),root=str(ROOT),device='cuda:0')
    run_suite(model,tok,rows,encoded,marker,cfg,1500 if '1500' in str(target) else 0,output)


if __name__=='__main__':
    main()
