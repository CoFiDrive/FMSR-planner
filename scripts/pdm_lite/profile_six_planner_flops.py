"""CPU operation-count audit of six planning cores, batch1 and 20 Euler steps.

Reports PyTorch FlopCounterMode's convolution/matrix-multiply/attention FLOPs,
with one multiply-add = two FLOPs. Uncounted operator types are exported.
This is not a full perception-to-control model count or latency benchmark.
"""
import csv
import gc
import importlib
import json
from pathlib import Path
import sys
from collections import Counter

ROOT=Path('/home/vcdc/fmsr')
sys.path[:0]=[str(ROOT/'3rd_party/carla_garage/team_code'),str(ROOT),
              str(ROOT/'3rd_party/CARLA_0915/PythonAPI/carla')]
import torch
import jsonpickle
import jsonpickle.ext.numpy as jnp
from torch.utils.flop_counter import FlopCounterMode, flop_registry
from torch.utils._python_dispatch import TorchDispatchMode
from torch.nn.attention import sdpa_kernel, SDPBackend
from config import GlobalConfig
from fmsr_interface import restore_trained_interface

OUT=ROOT/'outputs/analysis/model_parameter_counts_20261022'


class OperatorAudit(TorchDispatchMode):
    def __init__(self):
        super().__init__();self.calls=Counter()
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        self.calls[func._overloadpacket]+=1
        return func(*args,**(kwargs or {}))


def count(head,ego,bev,spatial):
    audit=OperatorAudit()
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH), audit, FlopCounterMode(display=False) as counter:
        prediction=head(ego_query=ego,bev_feature=bev,bev_spatial_shape=spatial)
    return dict(flops=counter.get_total_flops(),
        operation_flops={str(k):int(v) for k,v in counter.flop_counts['Global'].items()},
        uncounted_operator_calls={str(k):v for k,v in audit.calls.items() if k not in flop_registry},
        output_shape=list(prediction.shape))


def main():
    torch.set_num_threads(2);torch.manual_seed(0);jnp.register_handlers()
    # Avoid fused native MHA hiding its matmuls from the operation counter.
    torch.backends.mha.set_fastpath_enabled(False)
    counts=list(csv.DictReader((OUT/'parameter_counts.csv').open()))
    results=[];details={}
    for row in counts:
        checkpoint=Path(row['checkpoint']);folder=checkpoint.parent
        if row['run']=='R19':
            from fmsr.training.config_training import TrainingConfig
            config=TrainingConfig(loaded_config=json.loads((folder/'config.json').read_text()))
            module='fmsr.tfv6.diffusion_modules.model_diffusion_head_ddbm'
            poses=config.trajectory_sampling_num_poses;anchor=config.plan_anchor_path
            queries=config.num_route_points_prediction
            height=config.lidar_height_pixel//32;width=config.lidar_width_pixel//32
            prefix='planning_decoder.trajectory_head.'
        else:
            loaded=jsonpickle.decode((folder/'config.json').read_text())
            config=GlobalConfig();config.__dict__.update(loaded.__dict__)
            restore_trained_interface(config,loaded)
            modules={'baseline':'model_diffusion_head_ddbm','tie_cr_fd':'model_diffusion_head_tie_cr_fd',
                     'flow_only':'model_flow_matching_only'}
            module='fmsr.tfv6.diffusion_modules.'+modules[config.fmsr_trajectory_head]
            poses=config.fmsr_num_poses;anchor=config.fmsr_anchor_path
            queries=poses+int(config.fmsr_use_target_point)
            height=config.lidar_resolution_height//32;width=config.lidar_resolution_width//32
            prefix='fmsr_head.'
        head=getattr(importlib.import_module(module),'TrajectoryHead')(
            num_poses=poses,d_ffn=config.tf_d_ffn,d_model=config.tf_d_model,
            plan_anchor_path=anchor,config=config).eval()
        weights=torch.load(checkpoint,map_location='cpu',weights_only=True)
        weights=weights.get('state_dict',weights)
        weights={k.removeprefix('module.'):v for k,v in weights.items()}
        head.load_state_dict({k[len(prefix):]:v for k,v in weights.items() if k.startswith(prefix)},strict=True)
        del weights
        conv=next(m for m in head.modules() if isinstance(m,torch.nn.Conv2d))
        channels=conv.in_channels
        ego=torch.zeros(1,queries,config.tf_d_model)
        bev=torch.zeros(1,channels,height,width)
        native=count(head,ego,bev,(height,width))
        # Same input resolution provides an additional architecture comparison.
        common=count(head,ego,torch.zeros(1,channels,8,8),(8,8)) if (height,width)!=(8,8) else native
        result=dict(run=row['run'],method=row['method'],planning_head_M=float(row['planning_head_M']),
            full_model_M=float(row['full_model_M']),batch_size=1,
            flow_steps=int(config.flow_inference_steps),candidate_modes=int(head.plan_anchor.shape[0]),
            query_tokens=queries,BEV_channels=channels,BEV_height=height,BEV_width=width,
            planning_GFLOPs=native['flops']/1e9,planning_GMACs=native['flops']/2e9,
            planning_GFLOPs_at_8x8=common['flops']/1e9)
        results.append(result)
        details[row['run']]=dict(checkpoint=str(checkpoint),native=native,common_8x8=common)
        print(json.dumps(result),flush=True)
        del head,ego,bev;gc.collect()
    with (OUT/'planning_flops.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    (OUT/'flops_operator_audit.json').write_text(json.dumps(details,indent=2)+'\n')
    (OUT/'FLOPS_README.md').write_text('''# Planning-core operation count

Scope matches the preceding parameter table: trajectory_head / fmsr_head only.
Batch1, eval/no_grad, original checkpoints, 60 candidates, 20 Euler steps including
anchor classification. CPU operation counting only; no CUDA job or latency benchmark.
One multiply-add = 2 FLOPs; G = 1e9. GMACs are the counted multiply-add equivalents.
The full perception backbone, external planning context/adapters, controller and simulator
are excluded. CR training loss / Jacobian / SVD is not executed during inference.

Native spatial input shapes: FMSR BEV10x12 from320x384 LiDAR/stride32, Garage BEV8x8
from256x256/stride32. Garage8x8 is also verified by saved production input snapshots.
FMSR uses10 route queries/no speed token; Garage uses trained target-point query settings
and a speed token. All8x8 counts are also supplied as a separate comparison.

PyTorch FlopCounterMode counts convolution, matrix multiplication and decomposed attention
operations. MHA fastpath is disabled and SDPA math used to expose those operations, without
changing the algebra. Bias/normalization/activation/softmax/grid sampling/interpolation and
other elementwise/control operations are not included. Thus these are approximate major-op
GFLOPs (a partial operation count), not an exact sum of all floating-point instructions.
Every uncounted operator type and call count is retained in flops_operator_audit.json.
Input values are zeros because these heads have a fixed dense all-candidate execution path;
the tensor shapes and iteration count, not the selected mode, determine the main operation count.
''')


if __name__=='__main__':main()
