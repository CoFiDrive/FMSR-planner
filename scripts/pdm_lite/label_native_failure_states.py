"""Offline clean-prior reference on the student's exact tensors; no controls.

Run in a fresh process so the isolated Garage snapshot cannot replace modules
used by the production driving agent. Labels remain development candidates.
"""
import argparse
import gzip
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from fmsr_pdm_lite.fmsrpdm import bootstrap
import numpy as np
import torch
from scripts.pdm_lite.bounded_fm_recovery import config
from fmsr_pdm_lite.fmsrpdm.model import FMSRModel
from learned_teacher_labels import validate_prediction,disagreement,verify_training_directory


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection',type=Path,required=True)
    parser.add_argument('--evaluation',type=Path,required=True)
    parser.add_argument('--teacher',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=True
    torch.manual_seed(0)
    cp=json.loads((args.evaluation/'checkpoint_endpoint.json').read_text())
    if cp['entry_status']!='Finished' or cp['_checkpoint']['progress']!=[2,2]:
        raise ValueError('Both student routes must finish before offline labeling')
    cfg=config(args.teacher.parent/'config.json');teacher=FMSRModel(cfg).cuda().eval().requires_grad_(False)
    teacher.load_state_dict(torch.load(args.teacher,map_location='cpu',weights_only=True),strict=True)
    cases=[];candidate_keys=[];audit=[];repeatability=[]
    for record in cp['_checkpoint']['records']:
        folder=args.collection/'Recovery_Train'/record['save_name'].replace('_rep0_','_Rep0_')
        input_files=sorted((folder/'exact_inputs').glob('*.pt'))
        if not input_files:raise ValueError('No exact input frames')
        events_file=args.evaluation/'diagnostics'/record['save_name']/'events.jsonl'
        incidents=[json.loads(s) for s in events_file.read_text().splitlines()]
        first_collision=min((e['event_frame'] for e in incidents if 'COLLISION' in e['type']),default=float('inf'))
        all_events=[];gaps=[];near_gaps=[];valid=0;pre=0;static_reference=[];stationary=0
        for start in range(0,len(input_files),args.batch_size):
            files=input_files[start:start+args.batch_size];samples=[torch.load(p,map_location='cpu',weights_only=True) for p in files]
            keys=set(samples[0])
            if any(set(s)!=keys for s in samples):raise ValueError('Input key mismatch')
            batch={k:torch.cat([s[k] for s in samples]).cuda() for k in keys}
            with torch.no_grad():pred=teacher(**batch)
            for offset,p in enumerate(files):
                frame=int(p.stem);path=folder/'measurements'/f'{frame:04d}.json.gz'
                with gzip.open(path,'rt') as f:m=json.load(f)
                route,speed=validate_prediction(pred[2][offset].float().cpu().numpy()[:,:2],float(pred[1][offset,0]),10)
                student=np.asarray(m['student_route'],dtype=np.float32).reshape(10,2)
                gap=float(np.linalg.norm(route-student,axis=1).mean());near=float(np.linalg.norm(route[:3]-student[:3],axis=1).mean())
                before=m['simulation_frame']<first_collision
                # Numeric labels are valid; approval for training is separate.
                m.update(teacher_checkpoints=route.tolist(),route=route.tolist(),target_speed=speed,
                    label_valid=True,teacher_reference='clean_prior_epoch09',teacher_input_source='exact_saved_student_tensors',
                    training_eligible=False,pre_collision=before,teacher_control_applied=False,
                    brake=int(speed<.01),steer=0.,throttle=0.,teacher_control_available=False)
                with gzip.open(path,'wt') as f:json.dump(m,f,allow_nan=False)
                actual=abs(m['speed']);stationary=stationary+cfg.data_save_freq if actual<.2 else 0
                reasons=disagreement(student,m['student_target_speed'],route,speed,actual,stationary)
                all_events.append({'frame':frame,'simulation_frame':m['simulation_frame'],'student_step':m['student_step'],
                    'reasons':reasons,'label_valid':True,'training_eligible':False,'pre_collision':before})
                item={'route_id':record['route_id'],'frame':frame,'simulation_frame':m['simulation_frame'],
                    'pre_collision':before,'path_ADE_teacher_vs_student_m':gap,'near3_ADE_m':near,
                    'student_speed_mps':m['student_target_speed'],'teacher_speed_mps':speed,'actual_speed_mps':m['speed'],
                    'student_path':student.tolist(),'teacher_path':route.tolist(),'ego_world_position_m':m['ego_world_position_m'],
                    'ego_yaw_deg':m['ego_yaw_deg'],'reasons':reasons}
                audit.append(item);valid+=1
                if before:
                    pre+=1;gaps.append(gap);near_gaps.append(near)
                    if reasons:candidate_keys.append(f'Recovery_Train/{folder.name}/{frame:04d}')
        (folder/'recovery_events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in all_events))
        summary=json.loads((folder/'collection_summary.json').read_text());summary['valid_teacher_frames']=valid
        summary.update(reference_labelled=True,training_eligible=False,pre_collision_frames=pre)
        (folder/'collection_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        checks=verify_training_directory(folder)
        pre_indices=[e['frame'] for e in all_events if e['pre_collision']][-8:]
        if pre_indices:
            samples=[torch.load(folder/'exact_inputs'/f'{i:04d}.pt',map_location='cpu',weights_only=True) for i in pre_indices]
            repeated=[]
            for seed in (17,41):
                torch.manual_seed(seed)
                with torch.no_grad():
                    outputs=teacher(**{k:torch.cat([s[k] for s in samples]).cuda() for k in samples[0]})
                repeated.append((outputs[2].float().cpu(),outputs[1].float().cpu()))
            repeatability.append({'route_id':record['route_id'],'frames':pre_indices,'seeds':[17,41],
                'max_path_coordinate_difference_m':float((repeated[0][0]-repeated[1][0]).abs().max()),
                'max_speed_difference_mps':float((repeated[0][1]-repeated[1][1]).abs().max())})
        cases.append({'route_id':record['route_id'],'student_status':record['status'],'student_DS':record['scores']['score_composed'],
            'saved_frames':len(input_files),'pre_collision_frames':pre,'mean_pre_collision_teacher_student_ADE_m':float(np.mean(gaps)),
            'mean_pre_collision_near3_ADE_m':float(np.mean(near_gaps)),'first_collision_frame':first_collision,'roundtrip':checks})
        print('Labelled',record['route_id'],len(input_files),'frames',flush=True)
    args.report.mkdir(parents=True,exist_ok=True)
    (args.report/'same_state_reference.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in audit))
    (args.report/'candidate_pre_collision_frames.txt').write_text('\n'.join(candidate_keys)+'\n')
    result={'cases':cases,'candidate_disagreement_pre_collision_frames':len(candidate_keys),'teacher_checkpoint':str(args.teacher),
        'new_training_updates':0,'training_eligible':False,'scope':'same visited-state learned-reference predictions, not expert ground truth; benchmark development data'}
    (args.report/'label_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    (args.report/'teacher_repeatability.json').write_text(json.dumps(repeatability,indent=2)+'\n')


if __name__=='__main__':main()
