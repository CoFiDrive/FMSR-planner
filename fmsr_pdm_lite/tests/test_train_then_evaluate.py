import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys
from threading import Thread
import time

import pytest
import torch

from fmsrpdm import train_then_evaluate as job


def write_run(root):
    root.mkdir()
    (root/'run_config.json').write_text(json.dumps({'epochs':10,'generation_validation_samples':128,
                                                 'prediction_base':'tfpp_clean_prior_v1'}))
    (root/'config.json').write_text(json.dumps({'prediction_base':'tfpp_clean_prior_v1'}))
    (root/'provenance.json').write_text(json.dumps({'configuration_fingerprint':'config',
                                                 'asset_fingerprint':'assets'}))


def publish_epoch(root, epoch=9, complete=True):
    (root/f'epoch_{epoch:04d}.json').write_text(json.dumps({'epoch':epoch,'global_step':100,
                                                        'completed_full_epoch':complete}))
    (root/f'generation_epoch_{epoch:04d}.json').write_text(json.dumps({
        'samples':128,'scope':'predicted_anchor_full_sampling_no_label_input'}))
    (root/f'model_{epoch:04d}.pth').write_bytes(b'atomic model')
    torch.save({'epoch':epoch,'global_step':100,'config_fingerprint':'config','asset_fingerprint':'assets'},
               root/f'resume_{epoch:04d}.pt')


def test_ready_requires_target_epoch_full_completion_both_files_and_training_exit(tmp_path):
    root=tmp_path/'run';write_run(root)
    publish_epoch(root,8)
    assert not job.inspect_training(root,9,{})['ready']
    publish_epoch(root,9,False)
    assert not job.inspect_training(root,9,{})['ready']
    publish_epoch(root,9,True)
    companion=root/'resume_0009.pt';contents=companion.read_bytes();companion.unlink()
    assert not job.inspect_training(root,9,{})['ready']
    companion.write_bytes(contents)
    table={10:{'arguments':['python','-m','fmsrpdm.train','--output-dir',str(root)],'ppid':1},
           11:{'arguments':['python','-m','fmsrpdm.train','--output-dir=/another/run'],'ppid':1}}
    state=job.inspect_training(root,9,table)
    assert state['training_pids']==[10] and not state['ready']
    assert job.inspect_training(root,9,{})['ready']


def test_completion_companion_must_match_epoch_and_pinned_configuration(tmp_path):
    root=tmp_path/'run';write_run(root);publish_epoch(root)
    settings={'run_config_sha256':job.fingerprint(root/'run_config.json'),
              'config_sha256':job.fingerprint(root/'config.json')}
    assert job.verify_training_completion(root,9,settings)['epoch']==9
    torch.save({'epoch':8,'global_step':100,'config_fingerprint':'config','asset_fingerprint':'assets'},root/'resume_0009.pt')
    with pytest.raises(ValueError,match='companion'):
        job.verify_training_completion(root,9,settings)
    publish_epoch(root)
    (root/'config.json').write_text('{}')
    with pytest.raises(ValueError,match='configuration changed'):
        job.verify_training_completion(root,9,settings)


@pytest.mark.parametrize('record_metrics', [False, True])
def test_scheduled_job_waits_then_restarts_a_real_failed_watchdog(tmp_path,monkeypatch,record_metrics):
    root=tmp_path/'run';write_run(root)
    output=tmp_path/'evaluation'
    routes=tmp_path/'routes.xml';routes.write_text('<routes><route id="0"/><route id="1"/></routes>')
    package=tmp_path/'package';(package/'scripts').mkdir(parents=True)
    # An actual preflight subprocess checks that target weights are already published.
    preflight='''
import hashlib,json,os,pathlib
output=pathlib.Path(os.environ['EVALUATION_OUTPUT_DIR'])
model=pathlib.Path(os.environ['CHECKPOINT'])
assert model.name=='model_0009.pth' and model.is_file()
assert (model.parent/'resume_0009.pt').is_file()
routes=pathlib.Path(os.environ['ROUTES'])
hash_file=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(output/'source.json').write_text(json.dumps({'checkpoint':str(model),'route_count':2,
 'config_sha256':hash_file(model.parent/'config.json'),'routes_sha256':hash_file(routes),
 **({'analysis_recording': {'trace_every':1,'probe_every':200,'probe_max':4,
 'vehicle_state':'every_agent_tick','comfort_export_interval_seconds':0.1}}
 if os.environ.get('EVAL_CAPTURE_ALL_METRICS')=='1' else {})}))
'''
    import shlex
    (package/'scripts/evaluate.sh').write_text('#!/usr/bin/env bash\n'+shlex.quote(sys.executable)+
                                             " - <<'PY'\n"+preflight+'\nPY\n')
    # Replace only the subprocess command for the watchdog. The controller still
    # sees genuine early process exit, persistent partial scores, and a second run.
    fake_watchdog=tmp_path/'fake_watchdog.py'
    fake_watchdog.write_text('''
import json,pathlib,sys,os
output=pathlib.Path(sys.argv[1]);counter=output/'attempts.txt'
if 'analysis_recording' in json.loads((output/'auto_eval_job.json').read_text()):
 assert os.environ['EVAL_TRACE_EVERY']=='1' and os.environ['EVAL_PROBE_EVERY']=='200'
 assert os.environ['EVAL_PROBE_MAX']=='4' and os.environ['EVAL_METRIC_ROOT']==str(output)
attempt=int(counter.read_text())+1 if counter.exists() else 1
counter.write_text(str(attempt))
count=1 if attempt==1 else 2
(output/'checkpoint_endpoint.json').write_text(json.dumps({'entry_status':'Started' if count==1 else 'Finished',
 '_checkpoint':{'progress':[count,2],'records':[{'route_id':f'RouteScenario_{i}_rep0','status':'Completed'} for i in range(count)]}}))
sys.exit(1 if attempt==1 else 0)
''')
    monkeypatch.setattr(job,'PACKAGE_ROOT',package)
    monkeypatch.setattr(job,'WORKSPACE_ROOT',tmp_path)
    monkeypatch.setattr(job,'processes',lambda:{})
    actual_popen=job.subprocess.Popen
    def popen(command,*args,**kwargs):
        if 'fmsrpdm.evaluation_watchdog' in command:
            command=[sys.executable,str(fake_watchdog),str(output)]
        return actual_popen(command,*args,**kwargs)
    monkeypatch.setattr(job.subprocess,'Popen',popen)
    summaries=[]
    def summarize(directory):
        assert json.loads((directory/'checkpoint_endpoint.json').read_text())['_checkpoint']['progress']==[2,2]
        assert not job.lock_is_held(directory/'.evaluation.lock')
        summaries.append(directory)
        return {'benchmark_report':'benchmark.json','planning_report':'planning.json'}
    monkeypatch.setattr(job,'summarize_analysis',summarize)
    args=argparse.Namespace(run_dir=root,output=output,routes=routes,epoch=9,route_count=2,
        gpu=0,carla_port=8870,tm_port=8040,eval_timeout=120.,poll_seconds=.01,retry_seconds=.01,
        stale_seconds=900.,heartbeat_seconds=30.,rpc_timeout=5.,record_analysis_metrics=record_metrics)
    def publish_later():
        time.sleep(.05)
        assert not (output/'source.json').exists()
        publish_epoch(root)
    producer=Thread(target=publish_later);producer.start()
    original={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
    try:job.run_job(args)
    finally:
        producer.join()
        for s,handler in original.items():signal.signal(s,handler)
    assert (output/'attempts.txt').read_text()=='2'
    status=json.loads((output/'auto_eval_status.json').read_text())
    assert status['monitor_status']=='complete' and status['watchdog_starts']==2
    assert status['training']['target_epoch']==9
    assert json.loads((output/'training_completion.json').read_text())['epoch']==9
    assert len(summaries)==int(record_metrics)
    if record_metrics:
        assert 'analysis' in status

