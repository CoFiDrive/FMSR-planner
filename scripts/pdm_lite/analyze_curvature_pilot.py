"""Compare the fixed curvature-cap pilot against both existing policies."""
import json
from pathlib import Path
import statistics
from analyze_tfpp_speed_pilot import ROOT, DEFAULT, BASE, metrics, success, diagnostic

OUT=ROOT/'outputs/local_evaluation/bench2drive5_tie_spectral_epoch09_tfpp_curvature_20261008'
REPORT=ROOT/'outputs/analysis/fm_tfpp_curvature_closed_loop_20261008'


def main():
    p=OUT/'checkpoint_endpoint.json'
    if not p.exists():print('Waiting for first route');return
    data=json.loads(p.read_text());cp=data['_checkpoint'];rows=cp['records']
    rows=[r for r in rows if r.get('scores') and not r['status'].startswith("Failed - Agent couldn")]
    old={r['route_id']:r for r in json.loads((DEFAULT/'checkpoint_endpoint.json').read_text())['_checkpoint']['records']}
    fm={r['route_id']:r for r in json.loads(BASE.read_text())['_checkpoint']['records']}
    pairs=[]
    for r in rows:
        o=old[r['route_id']];b=fm[r['route_id']]
        pairs.append({'route_id':r['route_id'],'scenario':r['scenario_name'],'original_FM_DS':b['scores']['score_composed'],
            'tfpp_DS':o['scores']['score_composed'],'curvature_DS':r['scores']['score_composed'],
            'tfpp_success':success(o),'curvature_success':success(r),'status':r['status'],
            'infractions':{k:v for k,v in r['infractions'].items() if k!='min_speed_infractions'},
            'diagnostic':diagnostic(OUT/'diagnostics'/r['save_name']/'frames.jsonl'),
            'duration_game_s':r['meta']['duration_game'],'tfpp_duration_game_s':o['meta']['duration_game']})
    d={'progress':cp['progress'],'complete':cp['progress']==[5,5] and data['entry_status']=='Finished',
        'tfpp_same_routes':metrics([old[r['route_id']] for r in rows]),'curvature':metrics(rows),
        'original_FM_same_routes':metrics([fm[r['route_id']] for r in rows]),'pairs':pairs,
        'scope':'5 fixed targeted development routes; single curvature prior3m/s2, no parameter sweep'}
    (REPORT/'summary.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'progress':d['progress'],'complete':d['complete'],'tfpp':{k:v for k,v in d['tfpp_same_routes'].items() if k in ('DS','RC','SR_percent','success')},'curvature':{k:v for k,v in d['curvature'].items() if k in ('DS','RC','SR_percent','success')},'last':{k:v for k,v in pairs[-1].items() if k in ('route_id','scenario','tfpp_DS','curvature_DS','status')} if pairs else None},ensure_ascii=False))
    text=['# 曲率速度限制：5条定向闭环验证','',f"进度 {cp['progress'][0]}/5；已全部结束 {d['complete']}。",'',
      '使用原冻结TF++速度，按真实生成FM路径的三点曲率取最大值，以3 m/s²侧向加速度先验限制目标速度。没有训练、GT输入、参数搜索、控制器或安全覆盖修改。', '',
      '| 路线 | 原FM DS | TF++速度DS | 曲率限制DS | 曲率限制状态 | 仿真用时（TF++ / 限速）秒 |','| --- | ---: | ---: | ---: | --- | ---: |']
    for q in pairs:text.append(f"| {q['route_id']} | {q['original_FM_DS']:.4f} | {q['tfpp_DS']:.4f} | {q['curvature_DS']:.4f} | {q['status']} | {q['tfpp_duration_game_s']:.2f} / {q['duration_game_s']:.2f} |")
    text += ['',f"同路线TF++ DS {d['tfpp_same_routes'].get('DS',0):.4f}，曲率限制 DS {d['curvature'].get('DS',0):.4f}。",'',
      '离线128开发专家帧：25帧被限速；速度MAE从0.155733增至1.531768m/s，平均压低1.380217m/s。该规则明显偏保守，不适合据此直接作为默认。闭环用于检验碰撞与曲率/速度耦合，不能把速度MAE退化隐藏为收益。', '',
      '这是从已知失败中选取的单次开发验证；不能外推完整220条或作统计显著性结论。完整证据见summary.json与评估diagnostics目录。']
    (REPORT/'report.md').write_text('\n'.join(text)+'\n')


if __name__=='__main__':main()
