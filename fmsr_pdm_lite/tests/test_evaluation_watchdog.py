import json
from pathlib import Path

import pytest

from fmsrpdm.evaluation_watchdog import (
    associated_carla, evaluation_is_locked, evaluation_processes, inspect_checkpoint,
    FrameHeartbeat, carla_rpc_port, stop_matching_evaluators,
)


def test_native_probe_abort_is_reported_without_terminating_monitor(monkeypatch):
    from types import SimpleNamespace
    from fmsrpdm.evaluation_watchdog import FrameHeartbeat
    monkeypatch.setattr('fmsrpdm.evaluation_watchdog.subprocess.run',
                        lambda *a, **kw: SimpleNamespace(returncode=-6, stdout='', stderr='destroyed actor'))
    heartbeat=FrameHeartbeat(0.,30.,5.)
    heartbeat.poll({'12': {'arguments':['/x/CarlaUE4-Linux-Shipping','-carla-rpc-port=8950'],
                           'start_ticks':100}},31.)
    assert 'exited -6' in heartbeat.report(31.)['rpc_error']
    assert heartbeat.report(31.)['seconds_without_frame_progress']==31.


def write_checkpoint(path, statuses, entry='Finished', progress=None):
    records = [{'route_id': f'RouteScenario_{i}_rep0', 'status': status}
               for i, status in enumerate(statuses)]
    path.write_text(json.dumps({'entry_status': entry, '_checkpoint': {
        'records': records, 'progress': progress or [len(records), 2]}}))


def test_monitor_stops_for_all_scored_routes_including_driving_failures(tmp_path):
    path = tmp_path/'checkpoint.json'
    write_checkpoint(path, ['Completed', 'Failed - TickRuntime'])
    state, _ = inspect_checkpoint(path, ['RouteScenario_0_rep0', 'RouteScenario_1_rep0'])
    assert state['complete']


def test_monitor_retries_infrastructure_failures_and_incomplete_evaluation(tmp_path):
    path = tmp_path/'checkpoint.json'
    ids = ['RouteScenario_0_rep0', 'RouteScenario_1_rep0']
    state, payload = inspect_checkpoint(path, ids)
    assert not state['complete'] and payload is None
    write_checkpoint(path, ['Completed', 'Failed - Simulation crashed'])
    state, _ = inspect_checkpoint(path, ids)
    assert not state['complete'] and state['infrastructure_errors'] == [ids[1]]
    write_checkpoint(path, ['Completed'], 'Started')
    assert not inspect_checkpoint(path, ids)[0]['complete']


def test_monitor_rejects_mixed_route_checkpoints(tmp_path):
    path = tmp_path/'checkpoint.json'
    write_checkpoint(path, ['Completed', 'Completed'])
    with pytest.raises(ValueError):
        inspect_checkpoint(path, ['RouteScenario_2_rep0', 'RouteScenario_3_rep0'])


def test_old_failures_beyond_rewound_progress_are_pending_not_current_errors(tmp_path):
    path = tmp_path/'checkpoint.json'
    write_checkpoint(path, ['Failed - Agent crashed', 'Failed - Agent crashed'],
                     entry='Started', progress=[0, 2])
    ids = ['RouteScenario_0_rep0', 'RouteScenario_1_rep0']
    state, _ = inspect_checkpoint(path, ids)
    assert state['infrastructure_errors'] == []
    assert state['pending_retry_routes'] == ids
    assert not state['complete']


def test_monitor_does_not_start_a_duplicate_evaluation(tmp_path):
    import fcntl
    assert not evaluation_is_locked(tmp_path)
    with (tmp_path/'.evaluation.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert evaluation_is_locked(tmp_path)


def test_only_carla_descended_from_the_matching_evaluator_is_owned():
    target = Path('/evaluation/checkpoint_endpoint.json')
    table = {
        10: {'ppid': 1, 'arguments': ['/x/leaderboard_evaluator_v2.py', '--checkpoint='+str(target)]},
        11: {'ppid': 10, 'arguments': ['/bin/sh']},
        12: {'ppid': 11, 'arguments': ['/x/CarlaUE4-Linux-Shipping']},
        20: {'ppid': 1, 'arguments': ['/x/leaderboard_evaluator_v2.py', '--checkpoint=/other/checkpoint.json']},
        21: {'ppid': 20, 'arguments': ['/x/CarlaUE4-Linux-Shipping']},
    }
    evaluators = evaluation_processes(table, target)
    assert evaluators == {10}
    assert set(associated_carla(table, evaluators)) == {'12'}


def test_frame_heartbeat_tracks_simulation_progress_and_does_not_reset_on_rpc_errors():
    heartbeat=FrameHeartbeat(0.,30.,5.)
    heartbeat.observe((('12',100,8870,1000),),None,1.)
    heartbeat.observe((('12',100,8870,1000),),None,61.)
    assert heartbeat.report(61.)['seconds_without_frame_progress']==60.
    heartbeat.observe(None,'CARLA connection timeout',901.)
    assert heartbeat.report(901.)['seconds_without_frame_progress']==900.
    heartbeat.observe((('12',100,8870,1001),),None,902.)
    assert heartbeat.report(902.)['seconds_without_frame_progress']==0
    assert heartbeat.report(902.)['rpc_error'] is None
    assert carla_rpc_port({'arguments':['/x/CarlaUE4-Linux-Shipping','-carla-rpc-port=8872']})==8872


def test_stalled_restart_signals_only_identified_evaluator_and_escalates_after_grace(monkeypatch):
    import signal
    calls=[]
    monkeypatch.setattr('os.kill',lambda pid,signum:calls.append((pid,signum)))
    table={10:{'start_ticks':100},20:{'start_ticks':200}}
    stopped={}
    stop_matching_evaluators(table,{10},stopped,0.)
    stop_matching_evaluators(table,{10},stopped,15.)
    stop_matching_evaluators(table,{10},stopped,31.)
    assert calls==[(10,signal.SIGTERM),(10,signal.SIGKILL)]


def test_watchdog_resumes_after_a_real_subprocess_exits_early(tmp_path, monkeypatch):
    import argparse
    import shlex
    import signal
    import sys
    from fmsrpdm import evaluation_watchdog as watchdog

    package = tmp_path/'package'
    (package/'scripts').mkdir(parents=True)
    output = tmp_path/'results'
    output.mkdir()
    routes = tmp_path/'routes.xml'
    routes.write_text('<routes><route id="0"/><route id="1"/></routes>')
    (output/'source.json').write_text(json.dumps({
        'checkpoint': '/unused/model.pth', 'routes': str(routes), 'route_count': 2}))
    fake = '''
import json, os, pathlib, sys
output = pathlib.Path(os.environ['EVALUATION_OUTPUT_DIR'])
counter = output/'fake_attempts.txt'
attempt = int(counter.read_text())+1 if counter.exists() else 1
counter.write_text(str(attempt))
count = 1 if attempt == 1 else 2
records = [{'route_id':f'RouteScenario_{i}_rep0', 'status':'Completed'} for i in range(count)]
(output/'checkpoint_endpoint.json').write_text(json.dumps({
    'entry_status':'Started' if attempt == 1 else 'Finished',
    '_checkpoint': {'records':records, 'progress':[count,2]}}))
sys.exit(1 if attempt == 1 else 0)
'''
    (package/'scripts/evaluate.sh').write_text(
        '#!/usr/bin/env bash\n'+shlex.quote(sys.executable)+" - <<'PY'\n"+fake+'\nPY\n')
    monkeypatch.setattr(watchdog, 'PACKAGE_ROOT', package)
    monkeypatch.setattr(watchdog, 'WORKSPACE_ROOT', tmp_path)
    monkeypatch.setattr(watchdog, 'processes', lambda: {})
    args = argparse.Namespace(output=output, gpu=0, carla_port=8870, tm_port=8040,
                              retry_seconds=.01, poll_seconds=.01)
    original = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        watchdog.watch(args)
    finally:
        for s, handler in original.items():
            signal.signal(s, handler)
    assert (output/'fake_attempts.txt').read_text() == '2'
    status = json.loads((output/'watchdog_status.json').read_text())
    assert status['complete'] and status['monitor_status'] == 'complete'
    assert status['attempts'] == 2
    assert len(list(output.glob('checkpoint.before_watchdog_resume_*.json'))) == 1
