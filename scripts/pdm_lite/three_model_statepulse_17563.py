"""Six prospective clean/state-pulse runs; preserve every driving outcome."""
import os,sys,json,csv,subprocess
from pathlib import Path
from visual_case_campaign import ROOT,prepare,run,write
from fmsrpdm.evaluation_watchdog import inspect_checkpoint
MODELS=['FMSR','FM','FM_CR']
SPEC=dict(id='pulse_state_lateral_0.1',family='state',axis='lateral',amplitude=.1,tau=.4,mode='pulse',start_s=5.)
def main():
 out=Path(sys.argv[1]).resolve()
 if '--foreground' not in sys.argv:
  if out.exists():raise ValueError('Use a new directory')
  out.mkdir(parents=True)
  for c in ['clean','perturbed']:
   prepare(out/c,route_ids=['17563'],model_names=MODELS)
   spec=SPEC if c=='perturbed' else dict(id='clean',family='none',amplitude=0.,mode='continuous')
   write(out/c/'perturbation_job.json',dict(seed=0,route_conditions={'17563':spec}))
   for m in MODELS:
    f=out/c/m/'source.json';s=json.loads(f.read_text());s['analysis_recording'].update(probe_every=100,probe_max=40)
    s.update(save_actual_injection_arrays=True,route_perturbations={'17563':spec});write(f,s)
  write(out/'manifest.json',dict(route_ids=['17563'],models=MODELS,conditions=['clean','perturbed'],seed=0,total_route_runs=6,route_conditions={'17563':SPEC},
   protocol='One signed0.10m lateral translation of first10 route points at generation progress0.4 on first eligible planning tick after5s. FM tau0.4; diffusion t0.6. Speed token unchanged; original samplers/controllers retained. All outcomes retained.',
   weather='Original route17563 XML, fog_density50, daytime, no rain.',recording='Cameras every5 ticks; per-tick trajectories/control/events; probes every100 steps max40; actual before/after/delta injection arrays.'))
  with (out/'monitor.log').open('a') as log:
   p=subprocess.Popen([sys.executable,'-u',__file__,str(out),'--foreground'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  (out/'monitor.pid').write_text(str(p.pid)+'\n');print('PID',p.pid,'Output',out);return
 os.environ.update(VISUAL_SAVE_INJECTION_ARRAYS='1',VISUAL_PROBE_EVERY='100',VISUAL_PROBE_MAX='40')
 os.environ['PYTHONPATH']=os.pathsep.join([str(ROOT/'fmsr_pdm_lite'),str(ROOT),os.environ.get('PYTHONPATH','')])
 try:
  for c in ['clean','perturbed']:
   write(out/'status.json',dict(state='evaluating',condition=c,total_route_runs=6))
   os.environ['ROBUSTNESS_SPEC']=str(out/c/'perturbation_job.json');run(out/c)
  import numpy as np
  rows=[]
  for c in ['clean','perturbed']:
   for m in MODELS:
    root=out/c/m;state,payload=inspect_checkpoint(root/'checkpoint_endpoint.json',['RouteScenario_17563_rep0'])
    if not state['complete']:raise ValueError('Incomplete '+str(root))
    r=json.loads(payload)['_checkpoint']['records'][0];metric=root/'eval_metrics'/r['save_name'];diag=root/'diagnostics'/r['save_name']
    for f in [diag/'frames.jsonl',diag/'visual/frames.jsonl']:
     if not f.exists() or not f.stat().st_size:raise ValueError('Missing '+str(f))
    events=[e for line in (metric/'perturbations.jsonl').read_text().splitlines() for e in json.loads(line)['events']]
    arrays=list((metric/'injection_arrays').glob('*.npz'))
    if c=='clean' and events:raise ValueError('Unexpected clean injection')
    if c=='perturbed':
     if len(events)!=1 or len(arrays)!=1:raise ValueError('Expected exactly one audited pulse '+str(root))
     with np.load(arrays[0]) as data:
      delta=data['delta'];assert np.allclose(data['after']-data['before'],delta)
      assert np.allclose(np.abs(delta[...,:10,1]),.1,atol=1e-4) and np.allclose(delta[...,:10,0],0)
      assert np.allclose(delta[...,10:,:],0)
    rows.append(dict(model=m,condition=c,DS=r['scores']['score_composed'],RC=r['scores']['score_route'],status=r['status'],pulse_events=len(events),injection_samples=len(arrays),source=str(root/'checkpoint_endpoint.json'),infractions=json.dumps(r['infractions'])))
  with (out/'all_results.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  write(out/'recording_audit.json',dict(complete=True,route_runs=6,recordings=rows))
  write(out/'status.json',dict(state='complete',total_route_runs=6))
 except Exception as e:
  write(out/'status.json',dict(state='error',error=repr(e)));raise
if __name__=='__main__':main()
