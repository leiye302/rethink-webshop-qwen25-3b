"""FP32 native AdamW on CPU; do NOT step until the whole global batch ends."""
from __future__ import annotations
import torch

class CpuAdamWOffload:
    def __init__(self,model,*,lr=2e-6,betas=(.9,.95),weight_decay=.01,foreach=True):
        self.named=list(model.named_parameters())
        if not self.named or not all(p.requires_grad and p.dtype==torch.float32 for _,p in self.named):
            raise ValueError('Full trainable FP32 parameters are required')
        self.masters=[torch.nn.Parameter(p.detach().cpu().clone()) for _,p in self.named]
        self.optimizer=torch.optim.AdamW(self.masters,lr=lr,betas=betas,weight_decay=weight_decay,
                                         eps=1e-8,foreach=foreach)
        self.active=False;self.seen=set();self.hooks=[];self.updates=0
        for index,(_,parameter) in enumerate(self.named):
            self.hooks.append(parameter.register_post_accumulate_grad_hook(self._hook(index)))

    def _hook(self,index):
        def drain(parameter):
            if not self.active: raise RuntimeError('Backward outside an explicitly opened global batch')
            grad=parameter.grad
            if grad is None: raise RuntimeError('Missing leaf gradient')
            master=self.masters[index]
            copied=grad.detach().to(device='cpu',dtype=torch.float32)
            if master.grad is None: master.grad=copied.clone() if copied.data_ptr()==grad.data_ptr() else copied
            else: master.grad.add_(copied)
            parameter.grad=None
            self.seen.add(index)
        return drain

    def begin(self):
        if self.active: raise RuntimeError('Global batch is already open')
        self.optimizer.zero_grad(set_to_none=False)
        self.seen.clear();self.active=True

    def finish(self,clip=1.):
        if not self.active or len(self.seen)!=len(self.masters):
            raise RuntimeError('Incomplete full-parameter gradient batch')
        norm=torch.nn.utils.clip_grad_norm_(self.masters,clip,error_if_nonfinite=True,foreach=True)
        self.optimizer.step()
        with torch.no_grad():
            for (_,parameter),master in zip(self.named,self.masters): parameter.copy_(master)
        self.active=False;self.updates+=1
        return float(norm)

    def release_gradients(self):
        if self.active: raise RuntimeError('Cannot release an active batch')
        self.optimizer.zero_grad(set_to_none=True)

    def close(self):
        for h in self.hooks: h.remove()
        self.hooks.clear()
