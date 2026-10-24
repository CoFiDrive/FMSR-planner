"""Measured paired trajectory perturbations; exported data and publication figures."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from scripts.pdm_lite.compare_flow_robustness import load_head, fingerprint

OUT = Path('outputs/analysis/tie_perturbation_20261013')
ROOT = Path('outputs/local_evaluation/bench2drive220_fm_best_full_metrics_20261012')
WEIGHTS = {
    'FM': Path('outputs/local_training/flow_only_10epochs_seed0_20261008_143915/model_0009.pth'),
    'FM+CR': Path('outputs/local_training/tie_spectral_10epochs_seed0_20261006_223036/model_0009.pth'),
}


@torch.no_grad()
def run():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    heads = {k: load_head(v, 'cuda') for k, v in WEIGHTS.items()}
    ref = heads['FM']
    for h in heads.values():
        assert torch.equal(h.plan_anchor, ref.plan_anchor)
        assert torch.equal(h.std, ref.std)
    metrics = json.loads((ROOT/'analysis/benchmark_metrics.json').read_text())
    selected = {}
    for r in sorted(metrics['routes'], key=lambda r: r['route_id']):
        candidates = sorted((ROOT/'eval_metrics'/r['save_name']/'probes').glob('*.pt'))
        if candidates and r['scenario_name'] not in selected:
            selected[r['scenario_name']] = candidates[0]
    rows, sigmas = [], []
    amplitudes = [.02, .05, .10, .20]
    injections = [0, 8, 16]
    std = ref.std[0, 0, :10].float().cuda()
    start = time.perf_counter()
    for scene_index, (scenario, path) in enumerate(sorted(selected.items())):
        scene = torch.load(path, map_location='cpu', weights_only=False)
        ego, bev = (scene[k].float().cuda() for k in ('ego_query', 'bev_feature'))
        spatial = scene['bev_spatial_shape']
        anchors = ref.plan_anchor.float()[None]
        mode = int(ref._classify_flow_anchors(anchors, bev, spatial, ego)[-1].argmax(-1))
        anchor = anchors[:, mode:mode+1]
        times = ref.get_flow_schedule(device='cuda')

        def endpoint(h, x, t):
            n = len(x)
            return h.predict_flow_endpoint(x, t.expand(n), anchor.expand(n, -1, -1, -1),
                                           bev.expand(n, -1, -1, -1), spatial,
                                           ego.expand(n, -1, -1), None).float()

        def step(h, x, j):
            y = h._mask_flow_speed_padding(x+(times[j+1]-times[j])/(1-times[j])*(endpoint(h, x, times[j])-x))
            # The CR Jacobian holds the speed token fixed. Isolate that route block.
            y[..., 10:, :] = anchor[..., 10:, :]
            return y

        grids = {}
        for name, h in heads.items():
            grid = [anchor.clone()]
            for j in range(20):
                grid.append(step(h, grid[-1], j))
            grids[name] = grid
        for name, h in heads.items():
            for protocol in ('shared FM states', 'own rollout states'):
                grid = grids['FM'] if protocol == 'shared FM states' else grids[name]
                for injection in injections:
                    labels = [(axis, a, sign) for axis in (0, 1) for a in amplitudes for sign in (-1, 1)]
                    x = grid[injection].expand(len(labels)+1, -1, -1, -1).clone()
                    for i, (axis, a, sign) in enumerate(labels, 1):
                        x[i, 0, :10, axis] += sign*a
                    initial = torch.linalg.vector_norm((x[1:, 0, :10]-x[0, 0, :10])/std, dim=(-2, -1))
                    curves, physical = [], []
                    for j in range(injection, 21):
                        delta = x[1:, 0, :10]-x[0, 0, :10]
                        curves.append((torch.linalg.vector_norm(delta/std, dim=(-2, -1))/initial).cpu().numpy())
                        physical.append(torch.linalg.vector_norm(delta, dim=-1).mean(-1).cpu().numpy())
                        if j < 20:
                            x = step(h, x, j)
                    for i, (axis, amplitude, sign) in enumerate(labels):
                        rows.append(dict(scene=scenario, model=name, protocol=protocol, injection=injection,
                                         axis='longitudinal' if axis == 0 else 'lateral', amplitude_m=amplitude,
                                         sign=sign, tau=times[injection:].cpu().tolist(),
                                         gain=np.asarray(curves)[:, i].tolist(),
                                         mean_point_deviation_m=np.asarray(physical)[:, i].tolist()))
            # Identical FM rollout query states for a controlled endpoint-Jacobian comparison.
            for j in (0, 5, 10, 15, 19):
                x = grids['FM'][j]
                delta = torch.zeros(20, 1, 11, 2, device='cuda')
                for d in range(20):
                    delta[d, 0, d//2, d%2] = .1*std[d//2, d%2]
                pairs = torch.cat((x+delta, x-delta))
                f = endpoint(h, pairs, times[j])[:, 0, :10]
                jac = (((f[:20]-f[20:])/(.2*std)).flatten(1)).T
                sigma = float(torch.linalg.svdvals(jac)[0])
                alpha = float((times[j+1]-times[j])/(1-times[j]))
                update_sigma = float(torch.linalg.svdvals((1-alpha)*torch.eye(20, device='cuda')+alpha*jac)[0])
                sigmas.append(dict(scene=scenario, model=name, step=j, tau=float(times[j]),
                                   sigma=sigma, euler_sigma=update_sigma,
                                   threshold=[1.3, 1.2, 1.08, 1.0][min(3, j//5)]))
        print(f'{scene_index+1}/{len(selected)} {scenario}; elapsed {time.perf_counter()-start:.1f}s', flush=True)
        (OUT/'progress.json').write_text(json.dumps({'completed':scene_index+1, 'total':len(selected)}))
    report = dict(protocol='Fixed encoded FM conditions, common FM-selected anchor, fixed speed token; FP32 head replay',
                  sampling='First route by lexicographic route_id with probes per scenario, earliest saved probe; one snapshot per scenario',
                  limitations='Head-level route-block experiment; no GT accuracy, no speed coupling, no closed-loop guarantee; snapshots are from FM evaluation',
                  scene_count=len(selected), snapshots={k:str(v.resolve()) for k,v in selected.items()},
                  checkpoint_sha256={k:fingerprint(v) for k,v in WEIGHTS.items()},
                  device=torch.cuda.get_device_name(), elapsed_s=time.perf_counter()-start,
                  trials=rows, jacobians=sigmas)
    (OUT/'measurements.json').write_text(json.dumps(report, indent=2))
    plot(report)


def plot(report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Serif', 'font.size':10, 'pdf.fonttype':42,
                         'axes.spines.top':False, 'axes.spines.right':False})
    colors = {'FM':'#2878B5', 'FM+CR':'#D95319'}
    rng = np.random.default_rng(20261013)

    def aggregate(records, field):
        # Average +/- trials within a scene, then bootstrap independent scene units.
        scenes = sorted({r['scene'] for r in records})
        a = np.asarray([np.mean([r[field] for r in records if r['scene']==s], axis=0) for s in scenes])
        boot = a[rng.integers(0, len(a), (2000, len(a)))].mean(1)
        return a.mean(0), np.quantile(boot, .025, axis=0), np.quantile(boot, .975, axis=0)

    def save(fig, stem):
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(OUT/f'{stem}.{ext}', dpi=300, bbox_inches='tight')
        plt.close(fig)

    for protocol, stem in [('shared FM states','propagation_shared'), ('own rollout states','propagation_own')]:
        fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharey=True)
        for row, axis_name in enumerate(('lateral', 'longitudinal')):
            for col, injection in enumerate((0, 8, 16)):
                ax = axes[row,col]
                for model, color in colors.items():
                    records = [r for r in report['trials'] if r['protocol']==protocol and r['model']==model
                               and r['axis']==axis_name and r['injection']==injection and r['amplitude_m']==.1]
                    mean, low, high = aggregate(records, 'gain')
                    tau = records[0]['tau']
                    ax.plot(tau, mean, color=color, label=model, lw=2)
                    ax.fill_between(tau, low, high, color=color, alpha=.18)
                ax.axhline(1, color='.45', ls='--', lw=1)
                ax.set_title(f'{axis_name.capitalize()}, injection at $\\tau={injection/20:.1f}$')
                ax.set_xlabel('Sampling time $\\tau$'); ax.grid(alpha=.2)
                if col == 0: ax.set_ylabel('Normalized perturbation gain')
        axes[0,0].legend()
        fig.suptitle(f'{protocol.capitalize()} | 0.10 m displacement | {report["scene_count"]} scenes | 95% scene-bootstrap CI', y=1.02)
        fig.tight_layout(); save(fig, stem)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, axis_name in zip(axes, ('lateral', 'longitudinal')):
        for model, color in colors.items():
            means, lows, highs = [], [], []
            for amplitude in (.02, .05, .1, .2):
                records = [dict(r, terminal=r['mean_point_deviation_m'][-1]) for r in report['trials']
                           if r['protocol']=='shared FM states' and r['model']==model and r['axis']==axis_name
                           and r['injection']==8 and r['amplitude_m']==amplitude]
                mean, low, high = aggregate(records, 'terminal')
                means.append(mean); lows.append(low); highs.append(high)
            ax.plot([.02,.05,.1,.2], means, '-o', color=color, label=model)
            ax.fill_between([.02,.05,.1,.2], lows, highs, color=color, alpha=.18)
        ax.set(title=axis_name.capitalize(), xlabel='Injected displacement (m)', ylabel='Final mean point deviation (m)')
        ax.grid(alpha=.2); ax.legend()
    fig.suptitle('Perturbation amplitude sensitivity | shared states | injection at $\\tau=0.4$')
    fig.tight_layout(); save(fig, 'amplitude_sensitivity')
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for model, color in colors.items():
        for field, ax in [('sigma',axes[0]), ('euler_sigma',axes[1])]:
            means, lows, highs = [], [], []
            for step in (0,5,10,15,19):
                records = [r for r in report['jacobians'] if r['model']==model and r['step']==step]
                mean, low, high = aggregate(records, field)
                means.append(mean); lows.append(low); highs.append(high)
            axes[0 if field=='sigma' else 1].plot([0,.25,.5,.75,.95], means, '-o', color=color, label=model)
            ax.fill_between([0,.25,.5,.75,.95], lows, highs, color=color, alpha=.18)
        violations = [np.mean([r['sigma']>r['threshold'] for r in report['jacobians']
                               if r['model']==model and r['step']==s])*100 for s in (0,5,10,15,19)]
        axes[2].plot([0,.25,.5,.75,.95], violations, '-o', color=color, label=model)
    axes[0].step([0,.25,.5,.75,1], [1.3,1.2,1.08,1,1], where='post', color='.4', ls='--', label='CR threshold')
    axes[1].axhline(1, color='.4', ls='--')
    for ax, title, ylabel in zip(axes, ['Endpoint Jacobian','Euler update Jacobian','Endpoint threshold exceedance'],
                                ['Maximum singular value','Maximum singular value','Scenes exceeding threshold (%)']):
        ax.set(title=title, xlabel='Sampling time $\\tau$', ylabel=ylabel);ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Identical FM rollout query states | normalized route coordinates | fixed speed token')
    fig.tight_layout(); save(fig, 'jacobian_and_euler')
    summary = {}
    for model in colors:
        records = [r for r in report['trials'] if r['model']==model and r['protocol']=='shared FM states'
                   and r['amplitude_m']==.1 and r['injection']==8]
        summary[model] = {axis: float(np.mean([r['gain'][-1] for r in records if r['axis']==axis]))
                          for axis in ('lateral','longitudinal')}
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2))
    print('Summary', summary, flush=True)


if __name__ == '__main__':
    run()
