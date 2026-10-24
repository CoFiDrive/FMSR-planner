"""Descriptive offline comparison of requested historical best checkpoints."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scripts.pdm_lite.summarize_eval_metrics import summarize

ROOT = Path(__file__).resolve().parents[2]


def success(row):
    return row['scores']['score_route'] == 100 and not any(
        v for k,v in row['infractions'].items() if k != 'min_speed_infractions')


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    root = ROOT/'outputs/local_evaluation'
    chosen = dict(FM='bench2drive220_flow_only_10epochs_seed0_20261008_143915_epoch09',
                  FMSR='bench2drive220_author_v8_pdm_20261010_165733_epoch10',
                  FM_CR='bench2drive220_legacy_fm_cr_latest_merged_20261012')
    output = root/chosen['FM_CR']/'analysis/historical_best_comparison'
    output.mkdir(exist_ok=True)
    benchmark = ROOT/'3rd_party/Bench2Drive/tools/efficiency_smoothness_benchmark.py'
    data, metrics, record_maps = {}, {}, {}
    for model, directory in chosen.items():
        p = root/directory
        payload = json.loads((p/'checkpoint_endpoint.json').read_text())
        assert payload['_checkpoint']['progress'] == [220,220] and payload['entry_status']=='Finished'
        data[model] = payload
        records = payload['_checkpoint']['records']
        record_maps[model] = {r['route_id']:r for r in records}
        report = summarize(p/'checkpoint_endpoint.json', p/'eval_metrics', benchmark)
        global_scores = report['global_scores_mean']
        metrics[model] = dict(DS=global_scores['score_composed'], RC=global_scores['score_route'],
                             IS=global_scores['score_penalty'], SR=report['route_success_percent'],
                             Efficiency=report['efficiency_percent'], Comfort=report['comfort_percent'],
                             efficiency_routes=report['efficiency_route_count'],
                             comfort_routes=report['comfort_route_count'],
                             status_counts=dict(Counter(r['status'] for r in records)),
                             infraction_events={k:sum(len(r['infractions'].get(k,[])) for r in records)
                                                for k in records[0]['infractions'] if k!='min_speed_infractions'},
                             infraction_rates=payload['_checkpoint']['global_record']['infractions'])
        dump(output/f'{model}_benchmark_metrics.json',report)
    models = list(chosen)
    assert all(set(record_maps[m])==set(record_maps['FM_CR']) for m in models)
    comparisons = {}
    scenarios = sorted({r['scenario_name'] for r in record_maps['FM_CR'].values()})
    scenario_rows = []
    for scenario in scenarios:
        row = dict(scenario=scenario)
        for m in models:
            rr=[r for r in record_maps[m].values() if r['scenario_name']==scenario]
            assert len(rr)==5
            row[f'{m}_DS']=sum(r['scores']['score_composed'] for r in rr)/5
            row[f'{m}_SR']=100*sum(success(r) for r in rr)/5
        scenario_rows.append(row)
    for m in ('FM','FMSR'):
        pairs=[]
        for route_id,r in record_maps['FM_CR'].items():
            b=record_maps[m][route_id]
            assert r['scenario_name']==b['scenario_name'] and str(r['weather_id'])==str(b['weather_id'])
            pairs.append(dict(route_id=route_id,scenario=r['scenario_name'],baseline_DS=b['scores']['score_composed'],
                              FM_CR_DS=r['scores']['score_composed'],
                              delta_DS=r['scores']['score_composed']-b['scores']['score_composed'],
                              baseline_success=success(b),FM_CR_success=success(r)))
        comparisons[m]=dict(delta_DS=metrics['FM_CR']['DS']-metrics[m]['DS'],
                            delta_SR_pp=metrics['FM_CR']['SR']-metrics[m]['SR'],
                            DS_wins=sum(r['delta_DS']>1e-5 for r in pairs),
                            DS_ties=sum(abs(r['delta_DS'])<=1e-5 for r in pairs),
                            DS_losses=sum(r['delta_DS']<-1e-5 for r in pairs),
                            retained=sum(r['baseline_success'] and r['FM_CR_success'] for r in pairs),
                            rescued=sum(not r['baseline_success'] and r['FM_CR_success'] for r in pairs),
                            lost=sum(r['baseline_success'] and not r['FM_CR_success'] for r in pairs),
                            both_failed=sum(not r['baseline_success'] and not r['FM_CR_success'] for r in pairs),
                            routes=pairs)
    scope='Historical descriptive comparison: FM/FMSR highest-DS full runs; FM+CR latest-attempt failure-rerun composite. No CR-only causal or independent-seed significance claim.'
    result=dict(scope=scope,sources={m:str((root/d).resolve()) for m,d in chosen.items()},
                metrics=metrics,comparisons=comparisons,scenarios=scenario_rows)
    dump(output/'comparison.json',result)
    with (output/'overall_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['model','DS','RC','IS','SR','Efficiency','Comfort'])
        w.writeheader()
        for m in models:w.writerow(dict(model=m,**{k:metrics[m][k] for k in w.fieldnames[1:]}))
    with (output/'scenario_comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(scenario_rows[0]));w.writeheader();w.writerows(scenario_rows)
    colors=['#5470c6','#91cc75','#ee6666']
    labels=['FM (historical best)','FMSR (historical best)','FM+CR (rerun composite)']
    def save(fig,name):
        fig.savefig(output/f'{name}.png',dpi=180,bbox_inches='tight')
        fig.savefig(output/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for ax,key in zip(axes.flat,['DS','RC','IS','SR','Efficiency','Comfort']):
        vals=[metrics[m][key] for m in models]
        for i,v in enumerate(vals):
            if v is None:ax.text(i,.05,'Missing',ha='center',transform=ax.get_xaxis_transform())
            else:
                ax.bar(i,v,color=colors[i]);ax.annotate(f'{v:.3f}' if key=='IS' else f'{v:.2f}',(i,v),ha='center',va='bottom')
        ax.set_xticks(range(3),['FM','FMSR','FM+CR']);ax.set_title(key);ax.set_ylim(bottom=0)
    fig.suptitle('Historical results: FM+CR uses failure-rerun composition; protocols differ')
    fig.tight_layout();save(fig,'overall_metrics')
    matrix=np.array([[r[f'{m}_DS'] for m in models] for r in scenario_rows])
    fig,ax=plt.subplots(figsize=(8,15));im=ax.imshow(matrix,vmin=0,vmax=100,cmap='YlGnBu',aspect='auto')
    ax.set_yticks(range(len(scenarios)),[s.removesuffix('_1') for s in scenarios],fontsize=8)
    ax.set_xticks(range(3),['FM','FMSR','FM+CR composite'])
    for i in range(len(scenarios)):
        for j in range(3):ax.text(j,i,f'{matrix[i,j]:.1f}',ha='center',va='center',fontsize=8,color='white' if matrix[i,j]>65 else 'black')
    fig.colorbar(im,ax=ax,label='Mean DS (5 routes/scenario)');ax.set_title('44-scenario historical comparison');save(fig,'scenario_DS_heatmap')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,m in zip(axes,['FM','FMSR']):
        rr=comparisons[m]['routes'];ax.scatter([r['baseline_DS'] for r in rr],[r['FM_CR_DS'] for r in rr],alpha=.45)
        ax.plot([0,100],[0,100],'k--',lw=1);ax.set(xlabel=f'{m} DS',ylabel='FM+CR composite DS',title=f'Same 220 route IDs: {m} vs FM+CR',xlim=(-3,103),ylim=(-3,103))
    fig.tight_layout();save(fig,'paired_route_DS')
    keys=['collisions_vehicle','collisions_pedestrian','collisions_layout','red_light','vehicle_blocked','route_dev']
    fig,ax=plt.subplots(figsize=(11,5));x=np.arange(len(keys))
    for i,m in enumerate(models):
        ax.bar(x+(i-1)*.25,[metrics[m]['infraction_rates'][k] for k in keys],width=.25,color=colors[i],label=labels[i])
    ax.set_xticks(x,['Vehicle collision','Pedestrian collision','Static collision','Red light','Blocked','Deviation'],rotation=15)
    ax.set_ylabel('Events per driven km (official global statistic)');ax.legend(fontsize=8);fig.tight_layout();save(fig,'infraction_rates')
    lines=['# 历史最高版本离线对比','',
           'FM+CR 使用 DS=88.8349 / SR=75% 的最新复测合并结果；FM 和 FMSR 使用历史最高 DS 的完整 220 路线评测。',
           '这是历史结果的描述性比较，评测与训练条件并不完全相同；不能将全部差异归因于 CR。没有运行新评测。','',
           '| 模型 | DS | RC | IS | SR (%) | Efficiency (%) | Comfort (%) |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for m in models:
        vals=[metrics[m][k] for k in ['DS','RC','IS','SR','Efficiency','Comfort']]
        lines.append('| '+m+' | '+' | '.join('缺失' if v is None else f'{v:.4f}' for v in vals)+' |')
    lines+=['','最高分 FM 那次未保存 Comfort 所需逐帧数据；本报告没有借用其他 FM 评测的 Comfort。',
            'scenario_comparison.csv 包含全部 44 类场景；comparison.json 包含全部逐路线差异及成功/失败转换。',
            '图表提供 PNG 与 PDF：overall_metrics、scenario_DS_heatmap、paired_route_DS、infraction_rates。',
            '逐路线提升不等于单独 CR 的因果增益；FM+CR 的失败定向复测会影响分数分布。']
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(output=str(output),metrics=metrics,
                         comparisons={m:{k:v for k,v in comparisons[m].items() if k!='routes'} for m in comparisons}),
                     ensure_ascii=False,indent=2))


if __name__=='__main__':main()
