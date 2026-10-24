"""Final analysis subprocesses consume the durable telemetry exports."""
import json
from types import SimpleNamespace as NS

from scripts.pdm_lite.eval_metrics import EvalMetricLogger
from fmsrpdm.train_then_evaluate import summarize_analysis


def test_completed_metrics_generate_benchmark_and_planning_reports(tmp_path):
    class Actor:
        def get_world(self): return self
        def get_snapshot(self): return NS(frame=self.frame,timestamp=NS(elapsed_seconds=self.frame*.05))
        def get_velocity(self): return NS(x=1,y=0,z=0)
        def get_control(self): return NS(steer=0,throttle=0,brake=0)

    actor = Actor()
    metric = dict(acceleration=[0,0,0],angular_velocity=[0,0,0],forward_vector=[1,0,0],
                  right_vector=[0,1,0],location=[0,0,0],rotation=[0,0,0])
    logger = EvalMetricLogger(tmp_path,'config+route0')
    for step in range(81):
        actor.frame=step+1
        logger.record(step,actor.frame*.05,metric,actor)
        logger.record_plan(step,actor.frame*.05,{'phase':'inference',
                           'predicted_trajectory_ego_m':[[2.5,0],[5,0],[10,0]]},
                           actor.get_control(),actor.get_control())
    logger.close()
    (tmp_path/'checkpoint_endpoint.json').write_text(json.dumps({'entry_status':'Finished',
        '_checkpoint':{'progress':[1,1],'global_record':{'scores_mean':{'score_composed':100}},
          'records':[{'save_name':'route0','route_id':'RouteScenario_0_rep0','status':'Completed',
                      'scores':{'score_route':100},'infractions':{'min_speed_infractions':['Speed 100%']}}]}}))
    # A stale/incomplete derived export is reconstructed from the stream.
    (logger.directory/'metric_info.json').write_text('{}')
    reports = summarize_analysis(tmp_path)
    benchmark = json.loads((tmp_path/'analysis/benchmark_metrics.json').read_text())
    planning = json.loads((tmp_path/'analysis/planning_metrics.json').read_text())
    assert benchmark['comfort_complete'] and benchmark['comfort_percent']==100
    assert benchmark['efficiency_percent']==100 and benchmark['route_success_count']==1
    assert planning['routes'][0]['longitudinal_jerk_mps3']['max_abs']==0
    assert reports['planning_report'].endswith('planning_metrics.json')
