"""Outcome-adaptive case search, capped at3 rounds; preserve all outcomes."""
import sys,os,json,csv,subprocess,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from visual_case_campaign import ROOT,write

def main():
 out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
 write(out/'manifest.json',dict(route_id='23658',models=['FM','FM_CR'],seed=0,max_rounds=3,max_route_runs=12,
  stopping_rule='Stop after first complete round where SR_perturbed DS equals SR_clean DS within1e-6 and FM_perturbed DS < FM_clean DS; otherwise stop at3 rounds.',
  condition='BEV feature noise RMS ratio0.03 continuous from2s; fresh clean pair each round; unchanged seed0/checkpoints/weather/controller.',
  reporting='Adaptive outcome-selected case search, not an unbiased robustness estimate. Retain/report every round including nonmatching results. No claims of statistical significance.'))
 rows=[]
 try:
  for index in range(1,4):
   target=out/('round%02d'%index)
   write(out/'status.json',dict(state='evaluating',round=index,max_rounds=3))
   subprocess.run([sys.executable,str(ROOT/'scripts/pdm_lite/selected_perturbation_visual_campaign.py'),'--output',str(target),'--route-ids','23658'],check=True,cwd=ROOT)
   while True:
    f=target/'status.json'
    status=json.loads(f.read_text()) if f.exists() else {}
    if status.get('state')=='complete':break
    if status.get('state')=='error':raise RuntimeError(str(status))
    time.sleep(15)
   scores={}
   for cond in ['clean','perturbed']:
    for model in ['FM','FM_CR']:
     f=target/cond/model/'checkpoint_endpoint.json';r=json.loads(f.read_text())['_checkpoint']['records'][0]
     scores[cond,model]=r['scores']['score_composed']
     rows.append(dict(round=index,condition=cond,model=model,DS=r['scores']['score_composed'],RC=r['scores']['score_route'],status=r['status'],source=str(f)))
   matched=abs(scores['perturbed','FM_CR']-scores['clean','FM_CR'])<=1e-6 and scores['perturbed','FM']<scores['clean','FM']
   with (out/'all_rounds.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
   write(out/'search_summary.json',dict(rounds_completed=index,matched=matched,matched_round=index if matched else None,all_results=rows,
    limitation='Outcome-adaptive selection; matching scores are selected examples, not unbiased effect estimates.'))
   if matched:
    write(out/'status.json',dict(state='complete',reason='requested_pattern_observed',rounds_completed=index,matched_round=index));return
  write(out/'status.json',dict(state='complete',reason='three_round_limit_reached',rounds_completed=3,matched_round=None))
 except Exception as e:
  write(out/'status.json',dict(state='error',error=repr(e)));raise
if __name__=='__main__':main()
