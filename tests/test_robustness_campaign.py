import gzip
import json
from pathlib import Path
from scripts.pdm_lite.robustness_suite import archive,conditions,run
from scripts.pdm_lite.robustness_analysis import campaign_analysis


def test_archive_is_lossless_and_preserves_metadata(tmp_path):
    route=tmp_path/'eval_metrics'/'route';route.mkdir(parents=True)
    payload=b'{"sample": 1}\n'*100
    (route/'planning.jsonl').write_bytes(payload)
    (route/'metadata.json').write_text('{}')
    archive(tmp_path)
    assert gzip.open(route/'planning.jsonl.gz','rb').read()==payload
    assert not (route/'planning.jsonl').exists() and (route/'metadata.json').exists()
    assert len(json.loads((tmp_path/'archive.json').read_text())['files'])==1


def test_conditions_have_one_shared_clean_and_three_strengths():
    specs=conditions()
    assert len(specs)==25 and len({s['id'] for s in specs})==25
    assert sum(s['family']=='none' for s in specs)==1
    for family,axis in [('state','lateral'),('state','longitudinal'),('position','lateral'),('position','longitudinal')]:
        assert sum(s['family']==family and s.get('axis')==axis and s['mode']=='continuous' for s in specs)==3


def test_pilot_stop_marker_prevents_full_evaluation(tmp_path):
    (tmp_path/'STOP_BEFORE_FULL').touch()
    run(dict(output=str(tmp_path),data_root=str(tmp_path),pins={},
             jobs=[dict(id='full/clean/seed0/FM',phase='full')]))
    status=json.loads((tmp_path/'status.json').read_text())
    assert status['state']=='pilot_complete'
    assert not (tmp_path/'runs/full').exists()


def test_empty_campaign_reports_partial_without_fabricating_metrics(tmp_path):
    (tmp_path/'campaign.json').write_text(json.dumps({'jobs':[{'id':'a'}]}))
    campaign_analysis(tmp_path)
    report=json.loads((tmp_path/'analysis/summary.json').read_text())
    assert report['status']=='partial' and report['completed_jobs']==0
    assert not (tmp_path/'analysis/run_metrics.csv').exists()


def test_paired_advantage_compares_each_models_own_baseline(tmp_path):
    jobs=[]
    # Different clean scores: direct perturbed-score difference would be misleading.
    for model,clean_score,pert_score in [('FM',80.,70.),('FM_CR',90.,85.)]:
        for condition,score in [('clean',clean_score),('state',pert_score)]:
            job=dict(id=f'pilot/{condition}/{model}',phase='pilot',seed=0,model=model,
                     perturbation=dict(id=condition,family='none' if condition=='clean' else 'state',
                                       axis='lateral',amplitude=0. if condition=='clean' else .1,mode='continuous'))
            jobs.append(job);directory=tmp_path/'runs'/job['id'];(directory/'analysis').mkdir(parents=True)
            (directory/'complete.json').write_text('{}')
            route=dict(route_id='RouteScenario_1_rep0',scenario_name='Accident_1',scores={'score_composed':score},route_success=True,infractions={})
            report=dict(global_scores_mean=dict(score_composed=score,score_route=100,score_penalty=1),
                        routes=[route],route_count=1,route_success_percent=100,efficiency_percent=100,comfort_percent=20)
            behavior=dict(routes=[dict(route_id=route['route_id'],driven_distance_m=100,injection_eligible=True,injection_event_count=1)])
            (directory/'analysis/benchmark_metrics.json').write_text(json.dumps(report))
            (directory/'analysis/behavior_metrics.json').write_text(json.dumps(behavior))
    (tmp_path/'campaign.json').write_text(json.dumps({'jobs':jobs}))
    campaign_analysis(tmp_path)
    import csv
    with (tmp_path/'analysis/paired_CR_advantage.csv').open() as f:row=next(csv.DictReader(f))
    assert float(row['CR_degradation_advantage_DS'])==5.
    assert row['CI_low']==''  # One scenario cannot support an informative bootstrap interval.
    assert (tmp_path/'analysis/pilot_robustness_degradation.png').exists()
    assert (tmp_path/'analysis/pilot_safety_increments.pdf').exists()
