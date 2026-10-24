"""Verify actual production forward changes only returned speed for hybrid config."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.pdm_lite.bounded_fm_recovery import config,native_flow_model,loader,inputs,write_json
import torch


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=True
    torch.cuda.set_per_process_memory_fraction(.5)
    root=ROOT/'outputs/analysis/fm_tfpp_speed_closed_loop_20261007'
    cfg=config(root/'model/config.json')
    model=native_flow_model(cfg).to('cuda').eval().requires_grad_(False)
    model.load_state_dict(torch.load(root/'model/model_0009.pth',map_location='cpu',weights_only=True),strict=True)
    old=ROOT/'outputs/analysis/bounded_fm_recovery_20261007'
    _,batches=loader(cfg,old/'validation_frames.txt',4)
    expected=json.loads((ROOT/'outputs/analysis/bounded_speed_corrector_20261007/frozen_tfpp_speed_probe.json').read_text())['variants']['original_uncertainty_decoder']['predicted_speed_mps']
    paths_delta=0.;speeds=[]
    with torch.no_grad():
        for batch in batches:
            args=inputs(batch,'cuda')
            model.config.fmsr_speed_source='planner';original=model(**args)
            model.config.fmsr_speed_source='tfpp_uncertainty_v1';hybrid=model(**args)
            delta=float((original[2]-hybrid[2]).abs().max());paths_delta=max(paths_delta,delta)
            torch.testing.assert_close(original[2],hybrid[2],atol=0,rtol=0)
            speeds.extend(hybrid[1][:,0].cpu().tolist())
    torch.testing.assert_close(torch.tensor(speeds),torch.tensor(expected),atol=1e-5,rtol=1e-5)
    report={'frames':len(speeds),'path_max_abs_difference_m':paths_delta,
        'path_bitwise_equal_for_same_inputs':True,'returned_speed_matches_frozen_tfpp_probe':True,
        'no_parameter_or_solver_change':True,'speed_source':cfg.fmsr_speed_source}
    write_json(root/'production_forward_verification.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
