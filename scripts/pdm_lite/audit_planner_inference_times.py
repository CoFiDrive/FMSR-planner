"""Read recorded planning-head CUDA-event timings; never substitutes whole-model latency."""
import csv
import gzip
import json
from pathlib import Path
from collections import Counter
import numpy as np

ROOT=Path('/home/vcdc/fmsr')
OUT=ROOT/'outputs/analysis/planner_inference_time_audit_20261022'
RUN=ROOT/'outputs/local_evaluation/robustness_campaign_20261014/runs/pilot/clean/seed0'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    stats=[];provenance={}
    with (OUT/'recorded_head_times.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['model','route_id','step','nfe','generated_modes','gpu_ms','wall_submit_ms','retained_after_warmup'])
        writer.writeheader()
        for model in ['FM','FM_CR']:
            job=RUN/model
            checkpoint=json.loads((job/'checkpoint_endpoint.json').read_text())
            samples=[];all_samples=[];counts=Counter();heads=Counter();checkpoints=set();route_counts={}
            for record in checkpoint['_checkpoint']['records']:
                folder=job/'eval_metrics'/record['save_name']
                metadata=json.loads((folder/'metadata.json').read_text())
                for weight in metadata.get('checkpoints',[]):
                    checkpoints.add(weight['checkpoint'])
                path=folder/'planning.jsonl'
                if not path.exists():path=folder/'planning.jsonl.gz'
                valid=0;retained=0
                with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as stream:
                    for line in stream:
                        row=json.loads(line)
                        generation=row.get('policy',{}).get('generation',[])
                        for entry in generation:
                            if entry.get('planner_gpu_ms') is None:continue
                            ms=float(entry['planner_gpu_ms'])
                            if not np.isfinite(ms) or ms<=0:continue
                            valid+=1;keep=valid>10
                            all_samples.append(ms)
                            if keep:samples.append(ms);retained+=1
                            counts[entry.get('nfe')]+=1;heads[entry.get('head')]+=1
                            writer.writerow(dict(model=model,route_id=record['route_id'],step=row['step'],
                                nfe=entry.get('nfe'),generated_modes=entry.get('generated_mode_count'),
                                gpu_ms=ms,wall_submit_ms=entry.get('planner_wall_submit_ms'),
                                retained_after_warmup=int(keep)))
                route_counts[record['route_id']]=dict(all=valid,after_warmup=retained)
            for scope,values in [('all_recorded',all_samples),('exclude_first10_per_route',samples)]:
                array=np.asarray(values)
                stats.append(dict(model=model,scope=scope,route_count=len(route_counts),sample_count=len(array),
                    mean_ms=float(array.mean()),sample_SD_ms=float(array.std(ddof=1)),
                    median_ms=float(np.median(array)),p95_ms=float(np.percentile(array,95)),
                    min_ms=float(array.min()),max_ms=float(array.max())))
            provenance[model]=dict(source=str(job),weight_paths=sorted(checkpoints),nfe_counts=dict(counts),
                                    head_counts=dict(heads),route_sample_counts=route_counts)
    with (OUT/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(stats[0]));writer.writeheader();writer.writerows(stats)
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (OUT/'README.md').write_text('''# Recorded pure planning-head inference time audit

Metric: planner_gpu_ms from scripts/pdm_lite/planner_telemetry.py (CUDA events).
Boundary: model.py capture.scene() immediately before fmsr_head(...), capture.finish()
immediately after. Includes anchor/mode classification and iterative trajectory/speed-token
generation; excludes perception backbone, BEV/route-query adapters before capture, image/data
preprocessing, controller and simulation. Diagnostic instrumentation remains inside this boundary.
These are historical instrumented closed-loop measurements, not fresh isolated microbenchmarks.

Source: seed0 clean pilot, 44 routes per model. Stats pool recorded calls, so longer routes
contribute more samples. Both all samples and exclusion of first10 valid planning calls per route
are supplied. Sample SD describes per-call latency variation, not variation across training seeds.
Recorded nfe and head/weight provenance are in provenance.json.

The other four historical settings lack the same metric in their original evaluation artifacts.
FMSR inference_latency.csv model_planning_ms measures closed_loop_inference.forward(data=...)
on sensor input tensors and therefore includes the model encoder; it is not pure head latency.
Do not use duration_system/duration_game to estimate planning-head latency. A uniform new
head-only benchmark is needed for a six-method comparison; no new GPU work is started here
because other GPU jobs are active.
''')
    print(json.dumps(stats,indent=2))


if __name__=='__main__':main()
