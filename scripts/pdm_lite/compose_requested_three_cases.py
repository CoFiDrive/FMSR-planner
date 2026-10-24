"""Complete the missing clean FMSR recording and render requested comparison."""
import os,sys,json,subprocess,shutil
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from visual_case_campaign import ROOT,prepare,run,write
out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=True)
clean=out/'bridge_clean'
if not (clean/'manifest.json').exists():
 prepare(clean/'clean',route_ids=['2709'],model_names=['FMSR'])
 write(clean/'manifest.json',dict(route_ids=['2709']))
 write(clean/'clean/perturbation_job.json',dict(seed=0,route_conditions={'2709':dict(id='clean',family='none',component='query',amplitude=0.,mode='continuous',start_s=2.)}))
os.environ.update(ROBUSTNESS_SPEC=str(clean/'clean/perturbation_job.json'),VISUAL_SAVE_INJECTION_ARRAYS='1',VISUAL_PROBE_EVERY='100',VISUAL_PROBE_MAX='40')
os.environ['PYTHONPATH']=os.pathsep.join([str(ROOT/'fmsr_pdm_lite'),str(ROOT),os.environ.get('PYTHONPATH','')])
write(out/'status.json',dict(state='recording_clean_FMSR'))
run(clean/'clean')
base=ROOT/'outputs/local_evaluation'
old=base/'stronger_perturbation_visual_scan_20261021/strength_2p0'
bev=base/'bev23658_query2709_clean_perturbed_seed0_20261021_204235'
fm=base/'route2709_query_strengths_seed0_20261021_220933'
br=base/'fmsr_route2709_query_strengths_seed0_20261021_225927'
sources={}
for rid,camp in [('2164',old),('23658',bev)]:
 sources[rid]={c+'/'+m:dict(campaign=str(camp),condition=c,model=m) for c in ['clean','perturbed'] for m in ['FM','FM_CR']}
sources['2709']={
 'clean/FM':dict(campaign=str(clean),condition='clean',model='FMSR'),
 'perturbed/FM':dict(campaign=str(br),condition='query_0p1/perturbed',model='FMSR'),
 'clean/FM_CR':dict(campaign=str(fm),condition='clean',model='FM_CR'),
 'perturbed/FM_CR':dict(campaign=str(fm/'query_0p1'),condition='perturbed',model='FM_CR')}
write(out/'manifest.json',dict(route_ids=['2164','23658','2709'],case_sources=sources,
 display_models={'2164':{'FM':'FM','FM_CR':'FM+SR'},'23658':{'FM':'FM','FM_CR':'FM+SR'},'2709':{'FM':'FMSR','FM_CR':'FM+SR'}},
 column_titles={'2164':'State pulse 0.20 m','23658':'BEV noise 0.03','2709':'Query noise 0.1'},
 map_sources={'2164':str(base/'selected_perturbation_real_cases_20261021/clean'),'23658':str(bev/'clean'),'2709':str(fm/'clean')},
 protocol='Independent runs; all camera panels use actual scored records. Route2709 clean FM+SR DS60 includes collision; retained without substitution.'))
write(out/'status.json',dict(state='rendering'))
subprocess.run([sys.executable,str(ROOT/'scripts/pdm_lite/render_selected_perturbation_real_cases.py'),'--campaign',str(out)],check=True,cwd=ROOT)
fig=out/'figures'
(fig/'caption.txt').write_text('Three selected real-recording comparisons. Columns:2164 state lateral pulse0.20m at5s,23658 continuous BEV noise0.03 from2s,2709 continuous Query noise0.1 from2s. Rows:scene context,actual map paths,first-model clean,first-model perturbed,FM+SR clean,FM+SR perturbed. First model is FM in columns1/2 and FMSR in column3, as marked in panel badges and map legends. Route2709 FMSR clean newly recorded; other recordings reused with exact provenance in keyframe_selection.json and manifest.json. FM+SR clean on2709 has DS60 from collision and is retained. Independent traffic realizations; selected cases do not establish causal or universal superiority.\n')
dst=old/'figures';backup=dst/('backup_before_three_columns_'+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S'));backup.mkdir()
for f in fig.iterdir():
 target=dst/f.name
 if target.exists():shutil.copy2(target,backup/f.name)
 shutil.copy2(f,target)
shutil.copy2(out/'manifest.json',dst/'three_column_sources.json')
write(out/'status.json',dict(state='complete',updated_figure=str(dst/'four_selected_scenarios_real_effects.png'),backup=str(backup)))
print('Updated:',dst,flush=True)
