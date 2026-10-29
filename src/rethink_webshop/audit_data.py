"""All-row, real-tokenizer SFT audit without loading weights or using CUDA/API."""
import argparse
from pathlib import Path
import sys
from .common import ROOT,confined,atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokenizer',default='assets/base')
    a=p.parse_args()
    sys.path.insert(0,str(ROOT/'src/sft'))
    from train import load_config,verify_reference,validate_dataset,encode_all,digest
    from transformers import AutoTokenizer
    cfg=load_config(ROOT/'configs/sft.json')
    core,reference=verify_reference(ROOT,cfg)
    train,val,order,data=validate_dataset(ROOT,cfg,core)
    tok=AutoTokenizer.from_pretrained(confined(a.tokenizer),local_files_only=True)
    token=core.register_token(tok)
    trainenc=encode_all(train,tok,core,cfg['max_length'])
    valenc=encode_all(val,tok,core,cfg['max_length'])
    report={'passed':True,'weights_loaded':False,'cuda_used':False,'api_called':False,
            'model_revision':cfg['model_revision'],'marker':token,
            'train_count':len(trainenc),'val_count':len(valenc),'batches':len(order),
            'order_sha256':digest(order),'max_train_tokens':max(len(x['input_ids']) for x in trainenc),
            'max_val_tokens':max(len(x['input_ids']) for x in valenc),
            'all_joint_boundaries_match':True,'data':data,'reference':reference}
    atomic_json(ROOT/'runs/data_audit.json',report)
    print({k:v for k,v in report.items() if k not in ['data','reference']})


if __name__=='__main__':
    main()
