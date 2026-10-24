"""Plot six runs' first-ten-epoch training ADE/FDE, normalized by epoch 1.

Run: python scripts/pdm_lite/plot_six_training_ade_fde.py
These logged training diagnostics are not independent validation or closed-loop
metrics. Normalization does not eliminate dataset/implementation differences.
"""
import argparse
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plot_six_training_losses import OUT, SETTINGS, smooth_values, smooth_note, smooth_suffix


def save(fig, stem):
    for extension in ['png', 'pdf', 'svg']:
        path = OUT / (stem + '.' + extension)
        fig.savefig(path, dpi=600, facecolor='white')
        print(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smooth-alpha', type=float, default=0,
                        help='EMA previous-value weight, in [0, 1); applied after normalization.')
    parser.add_argument('--gaussian-sigma', type=float, default=0,
                        help='Gaussian sigma in epoch points, reflect boundaries; applied after normalization.')
    args = parser.parse_args()
    if not 0 <= args.smooth_alpha < 1:
        parser.error('--smooth-alpha must be in [0, 1).')
    if not np.isfinite(args.gaussian_sigma) or args.gaussian_sigma < 0:
        parser.error('--gaussian-sigma must be finite and nonnegative.')
    if args.smooth_alpha and args.gaussian_sigma:
        parser.error('Choose EMA or Gaussian smoothing, not both.')
    suffix = smooth_suffix(args.smooth_alpha, args.gaussian_sigma)
    with (OUT / 'epoch_loss_summary.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    data, exported = {}, []
    for run, label, _, _ in SETTINGS:
        records = sorted((r for r in rows if r['run'] == run and 1 <= int(r['epoch']) <= 10),
                         key=lambda r: int(r['epoch']))
        epochs = [int(r['epoch']) for r in records]
        assert epochs == list(range(1, 11)), (run, epochs)
        data[run] = {}
        for metric in ['ADE', 'FDE']:
            original = np.array([float(r['train_' + metric]) for r in records])
            assert np.isfinite(original).all() and original[0] > 0, (run, metric)
            normalized = original / original[0]
            assert normalized[0] == 1.0
            smoothed = smooth_values(normalized, args.smooth_alpha, args.gaussian_sigma)
            data[run][metric] = smoothed
            for e, v, n, s in zip(epochs, original, normalized, smoothed):
                record = dict(run=run, setting=label, epoch=e, metric=metric,
                                 original_training_error=float(v), first_epoch_error=float(original[0]),
                                 normalized_training_error=float(n))
                if args.smooth_alpha:
                    record.update(EMA_alpha=args.smooth_alpha, smoothed_display_value=float(s))
                if args.gaussian_sigma:
                    record.update(Gaussian_sigma=args.gaussian_sigma, boundary_mode='reflect',
                                  smoothed_display_value=float(s))
                exported.append(record)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'axes.labelsize': 13, 'xtick.color': 'black',
                         'ytick.color': 'black', 'axes.labelcolor': 'black',
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})

    def draw(ax, metric):
        for run, label, color, marker in SETTINGS:
            ax.plot(range(1, 11), data[run][metric], color=color, marker=marker,
                    markersize=5, linewidth=2.2, label=label)
        ax.set(xlabel='Epoch', ylabel=f'Normalized Training {metric} '
               + rf'($\mathrm{{{metric}}}_e / \mathrm{{{metric}}}_1$)',
               xlim=(.8, 10.2), ylim=(0, 1.05))
        ax.set_xticks(range(1, 11))
        ax.set_yticks(np.arange(0, 1.01, .2))
        ax.grid(color='#E6E6E6', linewidth=.9)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(1.1)
            spine.set_color('black')

    note = 'Training-log metrics divided by each run\'s first epoch mean; normalization does not remove setting differences.'
    for metric in ['ADE', 'FDE']:
        fig, ax = plt.subplots(figsize=(10, 6.5))
        fig.subplots_adjust(left=.10, right=.98, top=.96, bottom=.31)
        draw(ax, metric)
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -.17), ncol=2,
                  frameon=False, fontsize=10, columnspacing=1.6, handlelength=2.6)
        fig.text(.5, .025, note, ha='center', fontsize=8.5, color='#444444')
        smooth_note(fig, args.smooth_alpha, args.gaussian_sigma)
        save(fig, f'six_methods_normalized_training_{metric.lower()}_first10' + suffix)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.subplots_adjust(left=.07, right=.985, top=.96, bottom=.31, wspace=.23)
    for ax, metric in zip(axes, ['ADE', 'FDE']):
        draw(ax, metric)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.52, .075), ncol=2,
               frameon=False, fontsize=10, columnspacing=1.8, handlelength=2.6)
    fig.text(.5, .025, note, ha='center', fontsize=9, color='#444444')
    smooth_note(fig, args.smooth_alpha, args.gaussian_sigma)
    save(fig, 'six_methods_normalized_training_ade_fde_first10' + suffix)
    with (OUT / ('six_methods_normalized_training_ade_fde_first10' + suffix + '.csv')).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)


if __name__ == '__main__':
    main()
