"""One disposable CUDA DDP/BF16 update from local TF++; network is forbidden."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'3rd_party/carla_garage/team_code'),str(ROOT),str(ROOT/'3rd_party/CARLA_0915/PythonAPI/carla')]
import jsonpickle
import jsonpickle.ext.numpy as jnp
import requests
import torch
import torch.distributed as dist
from model import LidarCenterNet
from checkpoint_compatibility import configure_local_checkpoint_initialization,validate_image_encoder_checkpoint


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(0);jnp.register_handlers()
    torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=True
    cfg=jsonpickle.decode(a.config.read_text());cfg.fmsr_trajectory_head='flow_only';cfg.flow_contract_enable=False
    configure_local_checkpoint_initialization(cfg,str(a.checkpoint))
    with patch.object(requests.sessions.Session,'request',side_effect=AssertionError('Network request during local initialization')):
        model=LidarCenterNet(cfg)
        weights=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
        n=validate_image_encoder_checkpoint(model.state_dict(),weights)
        message=model.load_state_dict(weights,strict=False)
    unexpected_missing=[k for k in message.missing_keys if not k.startswith(('fmsr_head.','fmsr_tp_encoder.'))]
    if unexpected_missing:raise AssertionError(f'Missing base weights: {unexpected_missing[:10]}')
    for key,value in model.state_dict().items():
        if key.startswith('backbone.image_encoder.') and key in weights:
            torch.testing.assert_close(value,weights[key],atol=0,rtol=0)
    del weights
    model.cuda().freeze_for_fmsr();model.eval();model.fmsr_head.train()
    inputs={k:v.cuda() for k,v in torch.load(a.input,map_location='cpu',weights_only=True).items()}
    target=model.fmsr_head.plan_anchor[0:1].clone()
    before=model.fmsr_head.time_mlp[1].weight.detach().clone()
    with tempfile.TemporaryDirectory(prefix='flow_only_smoke_') as temporary:
        dist.init_process_group('nccl',init_method='file://'+temporary+'/rdzv',rank=0,world_size=1)
        try:
            ddp=torch.nn.parallel.DistributedDataParallel(model,device_ids=[0],find_unused_parameters=True,broadcast_buffers=False)
            optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=1e-4)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                output=ddp(**inputs,fmsr_route=target[:,:10],fmsr_target_speed=target[:,-1,0],return_fmsr_loss=True)
                loss=sum(output['losses'].values())
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward()
            grads=[v.grad for v in model.parameters() if v.requires_grad and v.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads):raise FloatingPointError('Nonfinite/missing gradients')
            optimizer.step()
            if torch.equal(before,model.fmsr_head.time_mlp[1].weight):raise AssertionError('Optimizer did not update the planner')
            result={'status':'passed','online_requests':0,'image_encoder_tensors_verified':n,'base_missing_keys':unexpected_missing,
                'cuda_ddp_bfloat16_optimizer_updates':1,'loss':float(loss.detach()),
                'checkpoint_written':False,'input_source':str(a.input),
                'target_source':'synthetic anchor target for optimizer smoke only; not a training experiment'}
            a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
        finally:dist.destroy_process_group()


if __name__=='__main__':main()
