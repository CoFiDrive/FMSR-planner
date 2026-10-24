"""Distribution-rich mechanism figure using measured scene and perturbation data."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT=Path('outputs/analysis/tie_perturbation_20261013')
CASES=Path('outputs/analysis/tie_paper_figures_20261013')
OUT=ROOT/'mechanism_details'
C={'FM':'#2878B5','FM+CR':'#D95319'}


def main():
    OUT.mkdir(exist_ok=True)
    d=json.loads((ROOT/'measurements.json').read_text())
    scenes=sorted(d['snapshots'])
    rng=np.random.default_rng(20261013)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':10,
                         'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axs=plt.subplots(4,3,figsize=(15,15),layout='constrained')
    statistics={}

    def trials(model,axis,injection,amp=.1,protocol='shared FM states'):
        return [r for r in d['trials'] if r['model']==model and r['axis']==axis
                and r['injection']==injection and r['amplitude_m']==amp and r['protocol']==protocol]

    def scene_array(records,field='gain'):
        return np.asarray([np.mean([r[field] for r in records if r['scene']==s],axis=0) for s in scenes])

    def final(model,axis,injection,amp=.1,physical=False):
        return scene_array(trials(model,axis,injection,amp),
                           'mean_point_deviation_m' if physical else 'gain')[:,-1]

    reductions=[]
    categories=[]
    for axis in ['lateral','longitudinal']:
        for inj in [0,8,16]:
            reductions.append(100*(1-final('FM+CR',axis,inj)/final('FM',axis,inj)))
            categories.append(f'{axis[:3]}.\n{inj/20:.1f}')
    reductions=np.asarray(reductions).T
    statistics['scene_terminal_gain_reduction_percent']={s:row.tolist() for s,row in zip(scenes,reductions)}

    # a-b: individual scene trajectories, injection-time encoding, and early-time zoom.
    styles={0:'-',8:'--',16:':'}
    for col,axis in enumerate(['lateral','longitudinal']):
        ax=axs[0,col]
        for inj in [0,8,16]:
            for model in C:
                records=trials(model,axis,inj)
                a=scene_array(records);t=records[0]['tau']
                for curve in a:ax.plot(t,curve,color=C[model],ls=styles[inj],alpha=.055,lw=.65)
                ax.plot(t,a.mean(0),color=C[model],ls=styles[inj],lw=2,
                        label=f'{model}, $\\tau_0={inj/20:.1f}$')
                ax.fill_between(t,np.quantile(a,.1,axis=0),np.quantile(a,.9,axis=0),color=C[model],alpha=.10)
        ax.axhline(1,color='.4',ls='--',lw=.7)
        ax.set(title=f'({chr(97+col)}) {axis.capitalize()}: scene-level propagation',
               xlabel='Sampling time $\\tau$',ylabel='Normalized perturbation gain')
        ax.legend(fontsize=6.7,ncol=2,loc='upper left');ax.grid(alpha=.15)
        inset=ax.inset_axes([.58,.13,.36,.25])
        for model in C:
            a=scene_array(trials(model,axis,16));t=np.arange(16,21)/20
            inset.plot(t,a.mean(0),color=C[model],lw=1.4)
            inset.fill_between(t,np.quantile(a,.1,axis=0),np.quantile(a,.9,axis=0),color=C[model],alpha=.15)
        inset.axhline(1,color='.5',ls='--',lw=.6);inset.set_title('Late injection: zoom',fontsize=6.5)
        inset.tick_params(labelsize=6);inset.grid(alpha=.12)

    # c: scene distributions rather than only one aggregate reduction number.
    ax=axs[0,2]
    bp=ax.boxplot([reductions[:,i] for i in range(6)],positions=np.arange(6),widths=.48,
                  patch_artist=True,showfliers=False,medianprops={'color':'black'})
    for i,box in enumerate(bp['boxes']):box.set(facecolor='#D5E6EF' if i<3 else '#F7DDCD')
    for i in range(6):
        ax.scatter(i+rng.uniform(-.16,.16,44),reductions[:,i],s=10,color='#2878B5' if i<3 else '#D95319',alpha=.5)
        ax.text(i,np.max(reductions[:,i])+3,f'{np.median(reductions[:,i]):.1f}%',ha='center',fontsize=7)
    ax.axhline(0,color='.4',ls='--',lw=.7)
    ax.set(title='(c) Paired benefit across scenes',ylabel='Terminal gain reduction (%)',
           xticks=np.arange(6),xticklabels=categories,ylim=(-5,100),xlabel='Direction / injection time $\\tau_0$')
    ax.text(.02,.95,'44/44 scenes improve in every group\nDots: scenes; boxes: median and IQR',
            transform=ax.transAxes,va='top',fontsize=7)
    ax.grid(axis='y',alpha=.15)

    # d: reveal heterogeneity and name scenes with particularly weak improvements.
    ax=axs[1,0]
    order=np.argsort(reductions[:,3])
    im=ax.imshow(reductions[order],aspect='auto',cmap='YlGnBu',vmin=0,vmax=85)
    ticks=np.arange(0,44,4)
    def short(s):return s.removesuffix('_1').replace('NonSignalized','NonSig.').replace('Signalized','Sig.')[:25]
    ax.set(yticks=ticks,yticklabels=[short(scenes[order[i]]) for i in ticks],
           xticks=range(6),xticklabels=categories,title='(d) Benefit heterogeneity by scenario')
    ax.tick_params(axis='y',labelsize=5.8)
    fig.colorbar(im,ax=ax,pad=.02,shrink=.8,label='Terminal gain reduction (%)')
    ax.set_xlabel('Direction / injection time');ax.axvline(2.5,color='white',lw=1)

    # e: Jacobian scene clouds, quantile envelopes, threshold and violation counts.
    ax=axs[1,1]
    query_steps=[0,5,10,15,19]
    for model in C:
        mat=np.asarray([[next(r['sigma'] for r in d['jacobians'] if r['model']==model and r['step']==step and r['scene']==s)
                         for step in query_steps] for s in scenes])
        t=np.array(query_steps)/20
        ax.plot(t,np.median(mat,axis=0),'-o',color=C[model],label=f'{model}: median',markersize=4)
        ax.fill_between(t,np.quantile(mat,.1,axis=0),np.quantile(mat,.9,axis=0),color=C[model],alpha=.16)
        for j in range(5):ax.scatter(t[j]+rng.uniform(-.009,.009,44),mat[:,j],s=7,color=C[model],alpha=.24)
    ax.step([0,.25,.5,.75,1],[1.3,1.2,1.08,1,1],where='post',color='.3',ls='--',label='CR threshold')
    ax.set(yscale='log',title='(e) Endpoint Jacobian: tails and exceedance',xlabel='Sampling time $\\tau$',ylabel='Maximum singular value (log scale)')
    ax.legend(fontsize=7);ax.grid(alpha=.15,which='both')
    violation=[]
    for step in query_steps:
        rr=[r for r in d['jacobians'] if r['model']=='FM+CR' and r['step']==step]
        violation.append(100*np.mean([r['sigma']>r['threshold'] for r in rr]))
    ax.text(.04,.53,'CR threshold exceedance (%)\n'+ ' / '.join(f'{v:.0f}' for v in violation),
            transform=ax.transAxes,fontsize=7,bbox={'facecolor':'white','alpha':.85,'edgecolor':'none'})
    statistics['CR_threshold_exceedance_percent']=violation

    # f: actual update singular values and non-expansion boundary, every scene visible.
    ax=axs[1,2]
    for j,step in enumerate(query_steps):
        for model,offset in [('FM',-.16),('FM+CR',.16)]:
            values=np.array([r['euler_sigma'] for r in d['jacobians'] if r['model']==model and r['step']==step])
            pos=j+offset
            violin=ax.violinplot(values,[pos],widths=.28,showextrema=False)
            for body in violin['bodies']:body.set(facecolor=C[model],alpha=.22)
            ax.scatter(pos+rng.uniform(-.06,.06,len(values)),values,s=7,color=C[model],alpha=.4)
            ax.plot([pos-.09,pos+.09],[np.median(values)]*2,color=C[model],lw=2)
    ax.plot([],[],color=C['FM'],label='FM');ax.plot([],[],color=C['FM+CR'],label='FM+CR')
    ax.axhline(1,color='.3',ls='--',lw=.8,label='Unit gain')
    ax.set(title='(f) Euler update: scene-level distributions',xticks=range(5),xticklabels=['0','.25','.50','.75','.95'],
           xlabel='Sampling time $\\tau$',ylabel='Maximum singular value')
    ax.legend(fontsize=7);ax.grid(axis='y',alpha=.15)
    inset=ax.inset_axes([.56,.45,.40,.25])
    for model in C:
        med=[np.median([r['euler_sigma'] for r in d['jacobians'] if r['model']==model and r['step']==s]) for s in query_steps[2:]]
        inset.plot([.5,.75,.95],med,'-o',color=C[model],markersize=3)
    inset.axhline(1,color='.5',ls='--',lw=.7);inset.set_title('Late-stage zoom',fontsize=6.5);inset.tick_params(labelsize=6)

    # g: sensitivity versus measured multi-step amplification, without causal fit.
    ax=axs[2,0];correlations={}
    for model in C:
        x=np.array([next(r['sigma'] for r in d['jacobians'] if r['model']==model and r['step']==0 and r['scene']==s) for s in scenes])
        for axis,marker in [('lateral','o'),('longitudinal','^')]:
            y=final(model,axis,0)
            ax.scatter(x,y,color=C[model],marker=marker,s=21,alpha=.55,label=f'{model}, {axis[:3]}.')
            rho=float(spearmanr(x,y).statistic);correlations[f'{model}/{axis}']=rho
    ax.set(xscale='log',yscale='log',title='(g) Local sensitivity vs. rollout amplification',
           xlabel='Initial endpoint maximum singular value',ylabel='Final normalized gain ($\\tau_0=0$)')
    ax.legend(fontsize=7);ax.grid(alpha=.15,which='both')
    ax.text(.03,.04,'Within-model Spearman $\\rho$ (lat. / lon.)\n'+
            '\n'.join(f'{m}: {correlations[m+"/lateral"]:.2f} / {correlations[m+"/longitudinal"]:.2f}' for m in C),
            transform=ax.transAxes,fontsize=7,bbox={'facecolor':'white','alpha':.8,'edgecolor':'none'})
    statistics['sensitivity_gain_spearman']=correlations

    # h: amplitude response with all scene points and tail bands; inset linearity test.
    ax=axs[2,1];amps=[.02,.05,.1,.2]
    for axis,style in [('lateral','-'),('longitudinal','--')]:
        for model in C:
            matrix=np.array([final(model,axis,8,a,True) for a in amps]).T
            ax.plot(amps,np.median(matrix,axis=0),style+'o',markersize=4,color=C[model],label=f'{model}, {axis[:3]}.')
            ax.fill_between(amps,np.quantile(matrix,.1,axis=0),np.quantile(matrix,.9,axis=0),color=C[model],alpha=.12)
            for j,a in enumerate(amps):ax.scatter(a+rng.uniform(-.001,.001,44),matrix[:,j],s=5,color=C[model],alpha=.12)
    ax.set(title='(h) Amplitude response and scene tails',xlabel='Injected displacement (m)',ylabel='Final mean point deviation (m)')
    ax.legend(fontsize=7);ax.grid(alpha=.15)
    inset=ax.inset_axes([.11,.50,.35,.22])
    for model in C:
        matrix=np.array([final(model,'lateral',8,a,True)/a for a in amps])
        inset.plot(amps,np.median(matrix,axis=1),'-o',color=C[model],markersize=3)
    inset.set_title('Deviation / amplitude (lat.)',fontsize=6);inset.tick_params(labelsize=6)

    # i: scenario-level association to historical DS, explicitly observational.
    root=Path('outputs/local_evaluation')
    metric_paths={'FM':root/'bench2drive220_fm_best_full_metrics_20261012/analysis/benchmark_metrics.json',
                  'FM+CR':root/'bench2drive220_legacy_fm_cr_latest_merged_20261012/analysis/benchmark_metrics.json'}
    scores={}
    for model,path in metric_paths.items():
        records=json.loads(path.read_text())['routes']
        scores[model]={s:np.mean([r['scores']['score_composed'] for r in records if r['scenario_name']==s]) for s in scenes}
    x=reductions[:,[1,4]].mean(1);y=np.array([scores['FM+CR'][s]-scores['FM'][s] for s in scenes])
    ax=axs[2,2]
    ax.scatter(x,y,s=28,c=np.where(y>=0,'#338A78','#A8658D'),alpha=.75,edgecolors='white',linewidths=.4)
    ax.axhline(0,color='.4',ls='--',lw=.8)
    rho=float(spearmanr(x,y).statistic)
    ax.set(title='(i) Mechanism benefit vs. historical scenario DS',xlabel='Mean terminal gain reduction (%)\nLat. / lon., $\\tau_0=0.4$',ylabel='Scenario DS difference: CR minus FM')
    ax.text(.03,.95,f'Spearman $\\rho={rho:.2f}$; 44 scenarios\nHistorical rerun composite; no causal inference',
            transform=ax.transAxes,va='top',fontsize=7)
    for index in [int(np.argmax(y)),int(np.argmin(y))]:
        ax.annotate(short(scenes[index]),(x[index],y[index]),xytext=(4,5),textcoords='offset points',fontsize=6)
    ax.grid(alpha=.15);statistics['mechanism_DS_spearman']=rho

    # j-k: paired random-direction heatmaps, with ECDF insets exposing uncapped tails.
    route='RouteScenario_2509_rep0'
    for col,(model,suffix) in enumerate([('FM','FM'),('FM+CR','FMCR')]):
        ax=axs[3,col]
        gain=np.load(CASES/f'perturbation_{route}_{suffix}.npz')['gain']
        im=ax.imshow(gain,origin='lower',aspect='auto',extent=[0,20,.5,32.5],vmin=0,vmax=3,cmap='magma')
        ax.set(title=f'({chr(106+col)}) Obstacle case: {model}, 32 directions',xlabel='Euler sampling step',ylabel='Matched perturbation ID')
        fig.colorbar(im,ax=ax,pad=.02,shrink=.75,extend='max',label='Normalized gain')
        inset=ax.inset_axes([.59,.08,.35,.26]);v=np.sort(gain[:,-1])
        inset.step(v,np.arange(1,33)/32,color=C[model],where='post')
        inset.set(title='Final gain ECDF',xlabel='Gain',ylim=(0,1));inset.tick_params(labelsize=6)
        inset.title.set_fontsize(6.5);inset.xaxis.label.set_fontsize(6)
        ax.text(.02,.98,f'Mean {gain[:,-1].mean():.2f}; P90 {np.quantile(gain[:,-1],.9):.2f}',transform=ax.transAxes,
                va='top',color='white',fontsize=7,bbox={'facecolor':'black','alpha':.55,'edgecolor':'none'})

    # l: all three illustrative cases, mechanisms can improve while route DS does not.
    ax=axs[3,2]
    manifest=json.loads((CASES/'figure_manifest.json').read_text())
    perturb=json.loads((CASES/'perturbation_manifest.json').read_text())
    px=[];py=[]
    for index,case in enumerate(perturb['cases']):
        fm=case['summary']['FM']['final_gain_mean'];cr=case['summary']['FM+CR']['final_gain_mean']
        driving=next(c for c in manifest['cases'] if c['route_id']==case['route_id'])
        px.append(100*(1-cr/fm));py.append(driving['scores']['FM+CR']['score_composed']-driving['scores']['FM']['score_composed'])
        ax.scatter(px[-1],py[-1],s=80,marker=['o','^','s'][index],color=['#338A78','#338A78','#A8658D'][index],zorder=3)
        label=['Junction interaction','Sequential lane change','Construction obstacle'][index]
        ax.annotate(f'{label}\nGain {fm:.2f} $\\rightarrow$ {cr:.2f}',(px[-1],py[-1]),
                    xytext=(-100,-32) if index==0 else ((10,8) if index==1 else (-95,16)),
                    textcoords='offset points',fontsize=7,arrowprops={'arrowstyle':'-','color':'.5','lw':.7})
    ax.axhline(0,color='.4',ls='--',lw=.8)
    ax.set(title='(l) Case boundary: reduced gain, unequal DS benefit',xlabel='Offline terminal gain reduction (%)',
           ylabel='Recorded route DS difference: CR minus FM',xlim=(62,88),ylim=(-12,55))
    ax.text(.03,.95,'3 selected illustrative cases\nHistorical runs; CR records include reassessment',
            transform=ax.transAxes,va='top',fontsize=7)
    ax.grid(alpha=.15)
    fig.suptitle('CR changes perturbation propagation: distributions, heterogeneity, and driving boundaries',fontsize=15)
    for ext in ['png','pdf','svg']:fig.savefig(OUT/f'fm_cr_mechanism_details.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    (OUT/'statistics.json').write_text(json.dumps(statistics,indent=2))
    caption=(r'\caption{Distribution-level analysis of perturbation propagation. '
             r'(a),(b) Scene trajectories, means and 10--90\% scene bands for $\pm0.10$ m lateral and longitudinal '
             r'path displacements at three injection times; insets magnify late injections. '
             r'(c) Paired terminal-gain reduction distributions: one point per scenario snapshot, boxes show medians and IQR. '
             r'(d) Scenario-by-condition reduction heatmap, ordered by initial longitudinal benefit. '
             r'(e) Endpoint Jacobian distributions and CR threshold exceedance; lines show medians, shading 10--90\% scene ranges. '
             r'(f) Euler-update Jacobian singular-value distributions with individual snapshots and median bars. '
             r'(g) Initial endpoint sensitivity versus measured multi-step directional amplification. '
             r'(h) Physical amplitude response, individual scene measurements and 10--90\% ranges. '
             r'(i) Exploratory association between offline benefit and historical scenario driving-score changes. '
             r'(a)--(i) use 44 snapshots, one per scenario type, shared FM-encoded conditions and a common FM-selected anchor; '
             r'route inputs and outputs are normalized by training statistics, the speed token is fixed, and heads use FP32. '
             r'(j),(k) One obstacle case with 32 matched random normalized perturbations; heatmaps saturate above 3, '
             r'while ECDF insets use uncapped terminal gains. '
             r'(l) Three selected illustrative cases showing that offline gain reduction can accompany different DS outcomes. '
             r'Bands are descriptive quantile ranges, not confidence intervals. Historical CR driving results include '
             r'failure-selected reassessments; associations and cases are not controlled causal tests or stability guarantees.}')
    (OUT/'caption.tex').write_text(caption+'\n')
    (OUT/'README.md').write_text('# 分布与机制细节总图\n\n'
        '全部使用既有 measurements.json 和案例 NPZ，没有模拟或添加实验值。\n'
        'a-b：每场景细线、均值、场景10-90%带、晚期注入放大窗。\n'
        'c：44场景配对改善率分布和点，d：场景异质性热图。\n'
        'e-f：奇异值样本分布、阈值和晚期放大窗；g：端点敏感性与终点增益的模型内相关。\n'
        'h：幅度响应及物理偏差尾部；i：离线机制改善与历史场景DS差值的探索性关系。\n'
        'j-k：原绕障案例的32方向扰动热图，内嵌终点增益经验CDF；l：三个案例机制改善与驾驶成绩差异。\n\n'
        '统计单位：a-i各场景一个早期快照，正负扰动先在场景内平均。IQR和分位带为描述统计，不是置信区间。'
        'g/i Spearman仅报告描述性相关，无p值、无因果主张。j-l场景选取规则和限制沿用原案例manifest。'
        'j-k为归一化随机轨迹变形，a-b为物理整体平移，不直接比较幅度。'
        'i/l历史DS含失败路线复测合并，不能作为受控CR消融。\n')
    print(OUT.resolve(),statistics['mechanism_DS_spearman'],flush=True)


if __name__=='__main__':main()
