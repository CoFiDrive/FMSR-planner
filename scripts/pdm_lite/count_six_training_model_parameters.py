"""Count registered parameters after exact key/shape checks against six checkpoints.

CPU only. M = 1,000,000 parameters. Frozen nn.Parameters are included;
registered buffers and optimizer state are excluded. No inference/training is run.
"""
import csv
import gc
import json
from pathlib import Path
import sys

ROOT=Path('/home/vcdc/fmsr')
sys.path[:0]=[str(ROOT/'3rd_party/carla_garage/team_code'),str(ROOT),
              str(ROOT/'3rd_party/CARLA_0915/PythonAPI/carla')]
import torch
import jsonpickle
import jsonpickle.ext.numpy as jnp
from config import GlobalConfig
from fmsr_interface import restore_trained_interface
from model import LidarCenterNet

OUT=ROOT/'outputs/analysis/model_parameter_counts_20261022'
IDS=['R19','R05','R04','R13','R17','R20']
LABELS=['FMSR flow baseline','PDM flow','PDM flow_fixed','FD CR, delayed',
        'FM','FM+CR, spectral Jacobian']


def main():
    torch.set_num_threads(2);jnp.register_handlers();OUT.mkdir(parents=True,exist_ok=True)
    settings=list(csv.DictReader((ROOT/'outputs/analysis/local_training_loss_summary_20261020/run_settings.csv').open(encoding='utf-8-sig')))
    rows=[]
    for rid,label in zip(IDS,LABELS):
        entry=next(r for r in settings if r['run']==rid)
        folder=ROOT/'outputs/local_training'/entry['directory']
        checkpoint=folder/'model_0009.pth'
        if not checkpoint.exists():
            available=sorted(folder.glob('model_*.pth'))
            if not available:available=sorted(folder.glob('ckpt/model_*.pth'))
            if not available:raise FileNotFoundError(checkpoint)
            checkpoint=available[-1]
        if rid=='R19':
            from fmsr.training.config_training import TrainingConfig
            from fmsr.tfv6.tfv6_fmsr import TFv6
            cfg=TrainingConfig(loaded_config=json.loads((folder/'config.json').read_text()))
            network=TFv6(device=torch.device('cpu'),config=cfg)
            head_prefix='planning_decoder.trajectory_head.'
        else:
            loaded=jsonpickle.decode((folder/'config.json').read_text())
            cfg=GlobalConfig();cfg.__dict__.update(loaded.__dict__)
            restore_trained_interface(cfg,loaded)
            # Only affects initialization/download; network shapes are checked exactly.
            cfg.image_encoder_pretrained=False
            network=LidarCenterNet(cfg)
            head_prefix='fmsr_head.'
        weights=torch.load(checkpoint,map_location='cpu',weights_only=True)
        weights=weights.get('state_dict',weights)
        weights={k.removeprefix('module.'):v for k,v in weights.items()}
        expected=network.state_dict()
        missing=sorted(set(expected)-set(weights));extra=sorted(set(weights)-set(expected))
        mismatch=[k for k in set(expected)&set(weights) if expected[k].shape!=weights[k].shape]
        if missing or extra or mismatch:
            raise ValueError('{} architecture mismatch: missing={}, extra={}, shapes={}'.format(rid,missing,extra,mismatch))
        parameters=dict(network.named_parameters())
        total=sum(p.numel() for p in parameters.values())
        head=sum(p.numel() for name,p in parameters.items() if name.startswith(head_prefix))
        buffers=sum(b.numel() for b in network.buffers())
        row=dict(run=rid,method=label,checkpoint=str(checkpoint),
            full_model_parameters=total,full_model_M=total/1e6,
            planning_head_parameters=head,planning_head_M=head/1e6,
            planning_head_prefix=head_prefix,
            remaining_model_M=(total-head)/1e6,
            registered_buffer_elements=buffers,checkpoint_key_shape_match=True)
        rows.append(row);print(json.dumps(row),flush=True)
        del network,weights,expected,parameters;gc.collect()
    with (OUT/'parameter_counts.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/'README.md').write_text('''# Model parameter counts

Counts use nn.Module.named_parameters() from the actual configured architecture, with
exact state_dict key/shape agreement checked against each recorded checkpoint.
Epoch09 is preferred; if pruned, an available checkpoint from the same run is used.
Parameter counts describe architecture size, not the optimization epoch.
M = 1,000,000 parameters, not file MB/MiB. Full model includes registered frozen
parameters such as anchors and masks; buffers and optimizer state are excluded.
Planning head means fmsr_head.* for Garage runs and
planning_decoder.trajectory_head.* for the FMSR run, excluding perception/external adapters.
This is architecture size, not the number of parameters unfrozen during training.
CR regularization is a training loss and need not add inference-network parameters.
CPU construction disables pretrained initialization/download without changing shapes.
The original six training labels are retained; they are not the relabeled plot aliases.
''')


if __name__=='__main__':main()
