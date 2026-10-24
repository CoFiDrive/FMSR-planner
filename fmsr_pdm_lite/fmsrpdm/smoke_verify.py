"""Check real PDM input, frozen checkpoint tensors, and 20-step inference."""
import argparse
import json
from pathlib import Path

import jsonpickle
import jsonpickle.ext.numpy as jnp
import torch
from torch.utils.data import default_collate

from .configuration import RunConfig
from .dataset import create_dataset
from .model import FMSRModel
from .train import tensor_inputs


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    jnp.register_handlers()
    run=RunConfig(**json.loads((args.run_dir/'run_config.json').read_text()))
    config=jsonpickle.decode((args.run_dir/'config.json').read_text())
    model=FMSRModel(config)
    weights=torch.load(args.run_dir/'model_smoke.pth',map_location='cpu',weights_only=True)
    model.load_state_dict(weights,strict=True)
    pretrained=torch.load(run.tfpp_checkpoint,map_location='cpu',weights_only=True)
    if 'state_dict' in pretrained:pretrained=pretrained['state_dict']
    pretrained={k.removeprefix('module.'):v for k,v in pretrained.items()}
    frozen=[k for k in weights if not k.startswith(('planner.','navigation_encoder.'))]
    changed=[k for k in frozen if not torch.equal(weights[k],pretrained[k])]
    if changed:raise AssertionError(f'Frozen perception changed: {changed[:8]}')
    model.train()
    assert not model.backbone.training and not model.join.training and model.planner.training
    assert all(not p.requires_grad for n,p in model.named_parameters()
               if not n.startswith(('planner.','navigation_encoder.')))
    model.eval()
    dataset=create_dataset(run,config,validation=True)
    sample=dataset[0]
    batch=default_collate([sample])
    inputs=tensor_inputs(batch,torch.device('cpu'))
    for key in ('return_loss','target_route','target_speed'):inputs.pop(key)
    with torch.inference_mode():prediction=model(**inputs,generator=torch.Generator().manual_seed(0))
    speed,route=prediction[1],prediction[2]
    assert speed.shape==(1,1) and route.shape==(1,10,2)
    assert torch.isfinite(route).all() and torch.isfinite(speed).all()
    assert 0<=float(speed[0,0])<=max(config.target_speeds)
    report={'status':'passed','frozen_tensors_checked':len(frozen),'changed_frozen_tensors':changed,
            'strict_checkpoint_reload':True,'inference_steps':config.step_num,
            'rgb_shape':list(batch['rgb'].shape),'lidar_shape':list(batch['lidar'].shape),
            'predicted_route_shape':list(route.shape),'predicted_target_speed_mps':float(speed[0,0]),
            'validation_frames':len(dataset),'validation_uses_labels_for_inference':False}
    (args.run_dir/'smoke_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
