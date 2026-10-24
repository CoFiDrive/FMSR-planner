"""Compare paths after matching the FM label sampling convention."""
import sys,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'3rd_party/carla_garage/team_code'),str(ROOT),str(ROOT/'3rd_party/CARLA_0915/PythonAPI/carla')]
import numpy as np
from route_geometry import prepare_route,smooth_route,sample_route_on_polyline


def main():
    root=ROOT/'outputs/analysis/fm_failure_state_teacher_20261008'
    rows=[json.loads(s) for s in (root/'same_state_reference.jsonl').read_text().splitlines()]
    cfg=json.loads((ROOT/'outputs/analysis/fm_tfpp_speed_closed_loop_20261007/model/config.json').read_text())
    audits=[]
    for r in rows:
        if not r['pre_collision']:continue
        student=np.asarray(r['student_path']);teacher=np.asarray(r['teacher_path'])
        matched=smooth_route(prepare_route(teacher,20),20)[:10]
        ordered=sample_route_on_polyline(teacher,10)
        aim=int(min(np.clip(cfg['lateral_speed_scale']*r['actual_speed_mps']*3.6+cfg['lateral_speed_offset'],24,105)/10-2,9))
        def heading(path):return math.degrees(math.atan2(path[aim,1],path[aim,0]))
        gap=(heading(matched)-heading(student)+180)%360-180
        audits.append(dict(route_id=r['route_id'],frame=r['frame'],simulation_frame=r['simulation_frame'],
            raw_ADE_m=r['path_ADE_teacher_vs_student_m'],matched_ADE_m=float(np.linalg.norm(matched-student,axis=1).mean()),
            matched_near3_ADE_m=float(np.linalg.norm(matched[:3]-student[:3],axis=1).mean()),
            lookahead_index=aim,student_heading_deg=heading(student),teacher_raw_heading_deg=heading(teacher),
            teacher_matched_heading_deg=heading(matched),matched_heading_gap_deg=gap,
            teacher_matched_path=matched.tolist(),teacher_ordered_path=ordered.tolist(),
            ordered_near3_ADE_m=float(np.linalg.norm(ordered[:3]-student[:3],axis=1).mean()),
            ordered_heading_gap_deg=(heading(ordered)-heading(student)+180)%360-180))
    result={'scope':'same-state reference disagreement, not path error to ground truth; teacher matched to first-point2.5m then1m spacing',
        'cases':[],'frames':audits}
    for id in ('23901','24041'):
        a=[r for r in audits if r['route_id']==f'RouteScenario_{id}_rep0'];critical=a[-8:]
        result['cases'].append({'route_id':f'RouteScenario_{id}_rep0','pre_collision_frames':len(a),
            'mean_matched_ADE_m':float(np.mean([r['matched_ADE_m'] for r in a])),
            'last8_mean_matched_near3_ADE_m':float(np.mean([r['matched_near3_ADE_m'] for r in critical])),
            'last8_max_abs_heading_gap_deg':max(abs(r['matched_heading_gap_deg']) for r in critical),
            'last8_max_abs_ordered_heading_gap_deg':max(abs(r['ordered_heading_gap_deg']) for r in critical),
            'last8_mean_ordered_near3_ADE_m':float(np.mean([r['ordered_near3_ADE_m'] for r in critical])),
            'example':max(critical,key=lambda r:abs(r['matched_heading_gap_deg']))})
    (root/'route_representation_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['cases'],indent=2))


if __name__=='__main__':main()
