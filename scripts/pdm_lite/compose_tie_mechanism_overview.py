"""Compose one mechanism overview from recorded experiments; no new measurements."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('outputs/analysis/tie_perturbation_20261013')
CASES = Path('outputs/analysis/tie_paper_figures_20261013')
OUT = ROOT/'mechanism_overview'
COLORS = {'FM':'#2878B5', 'FM+CR':'#D95319', 'FMSR':'#699B52'}


def main():
    OUT.mkdir(exist_ok=True)
    data = json.loads((ROOT/'measurements.json').read_text())
    rng = np.random.default_rng(137)
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'axes.titlesize':10,
                         'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(4,3,figsize=(12,12),layout='constrained')

    def ci(records, field):
        scenes = sorted({r['scene'] for r in records})
        values = np.asarray([np.mean([r[field] for r in records if r['scene']==s],axis=0) for s in scenes])
        bootstrap = values[rng.integers(0,len(values),(2000,len(values)))].mean(1)
        return values.mean(0), np.quantile(bootstrap,.025,axis=0), np.quantile(bootstrap,.975,axis=0)

    def line(ax,x,records,field,model,style='-',label=None):
        mean,low,high = ci(records,field)
        ax.plot(x,mean,style,color=COLORS[model],lw=1.8,label=label or model)
        ax.fill_between(x,low,high,color=COLORS[model],alpha=.16)

    for row,axis_name in enumerate(['lateral','longitudinal']):
        for col,injection in enumerate([0,8,16]):
            ax = axes[row,col]
            for model in ['FM','FM+CR']:
                records = [r for r in data['trials'] if r['model']==model and r['protocol']=='shared FM states'
                           and r['axis']==axis_name and r['injection']==injection and r['amplitude_m']==.1]
                line(ax,records[0]['tau'],records,'gain',model)
            ax.axhline(1,color='.4',ls='--',lw=.8)
            ax.set(title=f'({chr(97+row*3+col)}) {axis_name.capitalize()}, injection $\\tau={injection/20:.1f}$',
                   xlabel='Sampling time $\\tau$',ylabel='Normalized perturbation gain')
            ax.grid(alpha=.18);ax.legend(fontsize=8)
            ax.set_ylim(bottom=0)
    for col,field in enumerate(['sigma','euler_sigma']):
        ax=axes[2,col]
        for model in ['FM','FM+CR']:
            means,lows,highs=[],[],[]
            for step in [0,5,10,15,19]:
                mean,low,high=ci([r for r in data['jacobians'] if r['model']==model and r['step']==step],field)
                means.append(mean);lows.append(low);highs.append(high)
            ax.plot([0,.25,.5,.75,.95],means,'-o',color=COLORS[model],label=model,markersize=4)
            ax.fill_between([0,.25,.5,.75,.95],lows,highs,color=COLORS[model],alpha=.16)
        if col==0:
            ax.step([0,.25,.5,.75,1],[1.3,1.2,1.08,1,1],where='post',color='.4',ls='--',label='CR threshold')
        else:
            ax.axhline(1,color='.4',ls='--',lw=.8,label='Unit gain')
        ax.set(title=['(g) Endpoint sensitivity','(h) Euler single-step sensitivity'][col],
               xlabel='Sampling time $\\tau$',ylabel='Maximum singular value')
        ax.legend(fontsize=8);ax.grid(alpha=.18)
    ax=axes[2,2]
    for axis_name,style in [('lateral','-'),('longitudinal','--')]:
        for model in ['FM','FM+CR']:
            means,lows,highs=[],[],[]
            for amplitude in [.02,.05,.1,.2]:
                records=[dict(r,terminal=r['mean_point_deviation_m'][-1]) for r in data['trials']
                         if r['model']==model and r['protocol']=='shared FM states' and r['axis']==axis_name
                         and r['injection']==8 and r['amplitude_m']==amplitude]
                mean,low,high=ci(records,'terminal');means.append(mean);lows.append(low);highs.append(high)
            ax.plot([.02,.05,.1,.2],means,style+'o',markersize=4,color=COLORS[model],label=f'{model}, {axis_name[:3]}.')
            ax.fill_between([.02,.05,.1,.2],lows,highs,color=COLORS[model],alpha=.12)
    ax.set(title='(i) Perturbation amplitude sensitivity',xlabel='Injected displacement (m)',
           ylabel='Final mean point deviation (m)')
    ax.legend(fontsize=7);ax.grid(alpha=.18)
    route='RouteScenario_2509_rep0'
    finals={}
    for col,(model,suffix) in enumerate([('FM','FM'),('FM+CR','FMCR')]):
        ax=axes[3,col]
        gain=np.load(CASES/f'perturbation_{route}_{suffix}.npz')['gain']
        finals[model]=float(gain[:,-1].mean())
        im=ax.imshow(gain,origin='lower',aspect='auto',extent=[0,20,.5,32.5],vmin=0,vmax=3,cmap='magma')
        ax.set(title=f'({chr(106+col)}) Obstacle case: {model}',xlabel='Euler sampling step',ylabel='Random perturbation ID')
        fig.colorbar(im,ax=ax,label='Normalized gain',extend='max',shrink=.8,pad=.02)
    ax=axes[3,2]
    manifests=json.loads((CASES/'figure_manifest.json').read_text())
    case=next(c for c in manifests['cases'] if c['route_id']==route)
    models=['FM','FM+CR','FMSR']
    scores=[case['scores'][m]['score_composed'] for m in models]
    bars=ax.bar(models,scores,color=[COLORS[m] for m in models],width=.6)
    ax.bar_label(bars,fmt='%.0f',padding=3)
    ax.set(title='(l) Same route: recorded driving scores',ylabel='Route driving score',ylim=(0,125))
    ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    ax.text(.03,.93,f'Offline final gain: {finals["FM"]:.2f} $\\rightarrow$ {finals["FM+CR"]:.2f}\nFM vs. FM+CR route DS: 36 $\\rightarrow$ 36',
            transform=ax.transAxes,va='top',fontsize=8)
    fig.suptitle('CR and perturbation propagation: mechanism evidence and a driving boundary',fontsize=15)
    for ext in ['png','pdf','svg']:
        fig.savefig(OUT/f'fm_cr_mechanism_overview.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    caption=(r'\caption{Controlled offline perturbation analysis of FM and FM+CR. '
             r'(a)--(f) Propagation of $\pm0.10$ m lateral and longitudinal path displacements injected at '
             r'$\tau=0$, $0.4$, and $0.8$, evaluated on 44 snapshots (one per scenario type). '
             r'(g)--(h) Maximum singular values of the normalized route endpoint Jacobian and its Euler update Jacobian '
             r'at identical FM rollout query states. (i) Final physical path deviation versus perturbation amplitude at '
             r'$\tau=0.4$. Shading in (a)--(i) denotes 95\% scene-bootstrap intervals. '
             r'All these measurements use fixed FM-encoded conditions, a common FM-selected anchor, a fixed speed token, '
             r'and FP32 head inference; they do not test the complete closed-loop system. '
             r'(j)--(k) An obstacle-case diagnostic with 32 matched random path perturbations of normalized norm $0.1$, '
             r'injected at the initial sampling state; values above 3 saturate the heatmap. '
             r'(l) Historical route scores for the same obstacle case: reduced offline perturbation gain accompanies '
             r'unchanged FM versus FM+CR driving scores. The FM+CR route record belongs to a failure-route reassessment '
             r'composite; these driving records are illustrative rather than a controlled causal comparison.}')
    (OUT/'caption.tex').write_text(caption+'\n')
    (OUT/'README.md').write_text(
        '# 多子图机制总图\n\n'
        'a-f：44 类场景的公共状态扰动传播；g-h：端点与 Euler 更新 Jacobian；i：物理扰动幅度敏感性。\n\n'
        'j-k：ConstructionObstacle 路线 2509 的 32 个随机方向扰动热图；l：同一路线历史驾驶成绩，'
        'FM=36，FM+CR=36，FMSR=100。该案例说明内部扰动增益改善不必然带来路线成绩提升，'
        '并非 FM+CR 相对 FM 退化。历史闭环结果不是受控消融，FM+CR 来自复测合并。\n\n'
        'a-i 来源：../measurements.json，每类一个早期快照；j-l 来源：'
        '../../tie_paper_figures_20261013 的 NPZ 和 figure_manifest.json，案例取第 200 步快照。'
        'a-f 扰动为整体路径平移（m）；j-k 为归一化随机方向，幅度和状态不同，不直接比较数值。\n\n'
        'PNG 为 300 dpi；PDF/SVG 为矢量。caption.tex 提供完整英文图注。'
        '生成脚本：scripts/pdm_lite/compose_tie_mechanism_overview.py。\n')
    print(OUT.resolve())


if __name__=='__main__':
    main()
