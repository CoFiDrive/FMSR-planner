"""Export observed training scalars and plot task fitting separately from CR diagnostics."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/analysis/fm_cr_training_curves_20261019'
RUNS = {
    'FM': ROOT / 'outputs/local_training/flow_only_10epochs_seed0_20261008_143915',
    'FM+CR': ROOT / 'outputs/local_training/tie_spectral_10epochs_seed0_20261006_223036',
}
TAGS = ['metric/flow_fm_loss', 'metric/flow_endpoint_l1', 'metric/route_ade', 'metric/route_fde',
        'train_step/loss_fmsr_cls', 'metric/flow_contract_loss',
        'metric/flow_contract_sigma_max', 'metric/flow_contract_kappa_mean',
        'metric/flow_contract_success', 'metric/flow_contract_weight_effective']


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    epoch_rows, block_rows, inventories = [], [], []
    stored = {}
    for model, folder in RUNS.items():
        print('Reading', model, flush=True)
        events = EventAccumulator(str(folder), size_guidance={'scalars': 0})
        events.Reload()
        available = events.Tags()['scalars']
        cfg = json.loads((folder / 'config.json').read_text())
        run_info = dict(model=model, folder=str(folder), scalar_tags=available,
                        batch_size=cfg['batch_size'], lr=cfg['lr'], epochs=cfg['epochs'],
                        seed=cfg['seed'], freeze_backbone=cfg['freeze_backbone'],
                        independent_validation_tags=[t for t in available if t.startswith(('val/', 'validation/', 'test/'))])
        for tag in TAGS:
            if tag not in available:
                continue
            records = events.Scalars(tag)
            steps = np.array([r.step for r in records])
            values = np.array([r.value for r in records])
            assert np.isfinite(values).all(), (model, tag)
            if tag == 'metric/flow_fm_loss':
                assert len(records) == 92270 and steps[0] == 1 and steps[-1] == 92270
                run_info['training_steps'] = len(records)
                run_info['logged_wall_hours'] = (records[-1].wall_time-records[0].wall_time)/3600
            for kind, width, output in [('epoch', 9227, epoch_rows), ('block', 500, block_rows)]:
                indices = (steps-1)//width
                for index in np.unique(indices):
                    selected = values[indices == index]
                    selected_steps = steps[indices == index]
                    output.append(dict(model=model, metric=tag, bin_type=kind, bin_index=int(index),
                                       first_step=int(selected_steps[0]), last_step=int(selected_steps[-1]),
                                       mean_step=float(selected_steps.mean()), epoch=float(selected_steps.mean()/9227),
                                       count=len(selected), mean=float(selected.mean()),
                                       batch_SD=float(selected.std(ddof=1)) if len(selected)>1 else None))
            stored[model, tag] = (steps, values)
        for tag in ['loss_total', 'loss_fmsr_reg', 'loss_fmsr_cls']:
            for r in events.Scalars(tag):
                epoch_rows.append(dict(model=model, metric=tag, bin_type='logged_epoch', bin_index=r.step,
                                       first_step=None, last_step=None, mean_step=None, epoch=r.step+1,
                                       count=None, mean=r.value, batch_SD=None))
        inventories.append(run_info)
    write_csv('epoch_metrics.csv', epoch_rows)
    write_csv('500step_block_metrics.csv', block_rows)
    (OUT / 'audit.json').write_text(json.dumps(inventories, indent=2) + '\n')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    colors = {'FM': '#4C8997', 'FM+CR': '#C76870'}

    def line(ax, model, tag, color=None, label=None, linestyle='-'):
        selected = [r for r in block_rows if r['model'] == model and r['metric'] == tag]
        ax.plot([r['epoch'] for r in selected], [r['mean'] for r in selected],
                color=color or colors[model], label=label or model, linewidth=1.6, linestyle=linestyle)

    for model in RUNS:
        line(axes[0, 0], model, 'metric/flow_fm_loss')
        line(axes[0, 1], model, 'metric/route_ade')
    line(axes[1, 0], 'FM+CR', 'metric/flow_contract_loss', label='FM+CR: CR loss')
    line(axes[1, 1], 'FM+CR', 'metric/flow_contract_sigma_max', label='Sampled spectral gain')
    line(axes[1, 1], 'FM+CR', 'metric/flow_contract_kappa_mean', color='#777777',
         label='Sampled threshold mean', linestyle='--')
    axes[1, 1].axhline(1, color='#999999', linewidth=.8, linestyle=':')
    settings = [
        ('A  Shared flow-matching training objective', 'FM L1 loss (normalized)'),
        ('B  Training-batch trajectory diagnostic', 'Training ADE (m)'),
        ('C  Additional regularization objective', 'Unweighted CR loss'),
        ('D  Sampled regularization diagnostics', 'Gain / threshold'),
    ]
    for ax, (title, ylabel) in zip(axes.flat, settings):
        ax.set(title=title, xlabel='Training epoch', ylabel=ylabel, xlim=(0, 10))
        ax.axvline(2, color='#aaaaaa', linestyle='--', linewidth=.8)
        ax.axvspan(2, 3, color='#f1e5d6', alpha=.4, zorder=-1)
        ax.grid(alpha=.18)
        ax.legend(frameon=False)
    axes[0, 0].set_yscale('log')
    fig.supxlabel('500-step block means; single training seed. Dashed vertical line: CR begins after 18,454 steps; ramp spans epochs 2-3.\n'
                  'Training diagnostics only: these are not held-out validation or closed-loop metrics.', fontsize=9)
    for extension in ['png', 'pdf']:
        fig.savefig(OUT / f'fm_cr_training_diagnostics.{extension}', dpi=600)
    plt.close(fig)
    final = [r for r in epoch_rows if r['bin_index'] == 9 and r['bin_type'] in ['epoch', 'logged_epoch']]
    for r in final:
        print(r['model'], r['metric'], r['mean'])
    report = ['# FM / FM+CR training-log audit', '',
              'Both runs: 10 epochs, 92,270 steps, 9,227 batches/epoch, batch size 32, LR 1e-4, seed0, frozen backbone.',
              'The configuration differences are primarily the flow head and CR settings; config alone does not verify every source-code or initialization difference.',
              'CR delay=18,454 steps, ramp=9,227 steps, maximum weight=0.01, regularization interval=4.',
              'No separately named validation/test scalar series were found. ADE/FDE shown are training-batch diagnostics.',
              'FM-only and FM+CR total/regression objectives differ, so total-loss magnitude is not a fair common-objective comparison.',
              'CR samples only some training steps; aggregate only recorded samples, without zero-filling missing steps.',
              'A near-zero effective weight on the last step does not mean CR never ran: CR is sampled every four steps.',
              'Curves use non-overlapping 500-step bins; batch_SD in CSV describes batch variation, not training-seed uncertainty.',
              'Positive state-response plots do not by themselves establish contraction; sampled gains and threshold compliance must also be examined.', '',
              '| Model | Metric | Final epoch mean |', '|---|---|---:|']
    for r in final:
        report.append(f'| {r["model"]} | {r["metric"]} | {r["mean"]:.7f} |')
    report += ['', '## Primary references checked', '',
               '- Validity Learning on Failures, Figures 3-4: https://arxiv.org/html/2406.01544',
               '- DriveTransformer, Appendix F: https://arxiv.org/html/2503.07656v1#A6',
               '- Bench2DriveZoo releases configurations, weights, evaluation JSON: https://github.com/Thinklab-SJTU/Bench2DriveZoo',
               '- Public weight listing: https://huggingface.co/rethinklab/Bench2DriveZoo/tree/main',
               '- Paper figures and final-loss tables are not equivalent to released raw TensorBoard/CSV logs.']
    (OUT / 'REPORT.md').write_text('\n'.join(report) + '\n')
    print('Saved', OUT, flush=True)


if __name__ == '__main__':
    main()
