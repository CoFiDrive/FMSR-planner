"""Validate actual captured labels through the production loader, no training."""
import argparse
import gzip
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'3rd_party/carla_garage/team_code'),str(ROOT),str(ROOT/'3rd_party/CARLA_0915/PythonAPI/carla')]
import jsonpickle
import jsonpickle.ext.numpy as jnp
import numpy as np
from data import CARLA_Data


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--collection',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--report',type=Path,required=True);a=parser.parse_args()
    jnp.register_handlers();cfg=jsonpickle.decode(a.config.read_text())
    cfg.fmsr_recovery_root=str(a.collection.resolve());cfg.augment=cfg.use_color_aug=cfg.lidar_aug_prob=0
    cfg.fmsr_frame_manifest='';cfg.fmsr_recovery_manifest=str((a.report/'loader_verification_only_frames.txt').resolve())
    keys=[]
    for route in sorted((a.collection/'Recovery_Train').iterdir()):
        paths=sorted((route/'measurements').glob('*.json.gz'));indices=sorted(set((min(10,len(paths)-1),len(paths)//2,len(paths)-1)))
        keys.extend(f'Recovery_Train/{route.name}/{i:04d}' for i in indices)
    Path(cfg.fmsr_recovery_manifest).write_text('\n'.join(keys)+'\n')
    dataset=CARLA_Data([str(a.collection/'Recovery_Train')],cfg);checks=[]
    if len(dataset)!=len(keys):raise ValueError(f'Native loader found {len(dataset)} instead of {len(keys)} samples')
    for i in range(len(dataset)):
        sample=dataset[i];frame=int(dataset.sample_start[i]);folder=Path(dataset.measurements[i][0].decode())
        with gzip.open(folder/f'{frame:04d}.json.gz','rt') as f:m=json.load(f)
        np.testing.assert_allclose(sample['route'][:10],m['teacher_checkpoints'],rtol=1e-5,atol=1e-5)
        np.testing.assert_allclose(sample['target_speed_value'],m['target_speed'],rtol=1e-5)
        if not np.isfinite(sample['rgb']).all() or not np.isfinite(sample['lidar']).all():raise ValueError('Nonfinite native tensors')
        checks.append({'frame':frame,'measurement_folder':str(folder),'route_preserved':True,'speed_preserved':True,
            'rgb_shape':list(sample['rgb'].shape),'lidar_shape':list(sample['lidar'].shape),'training_eligible':m['training_eligible']})
    report={'loader_source':'production carla_garage/team_code/data.py','training_updates':0,'samples':checks,
        'scope':'verification-only manifest, not a training manifest'}
    (a.report/'native_loader_verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
