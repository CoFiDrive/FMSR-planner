"""Overlay six runs' first ten epoch total losses; --normalize plots L_e/L_1."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/analysis/local_training_loss_summary_20261020'
SETTINGS = [
    ('R19', 'FMSR flow baseline (BS32, LR 3e-4)', '#8172B3', 'v'),
    ('R05', 'PDM flow (BS16, LR 3e-4)', '#55A868', 's'),
    ('R04', 'PDM flow_fixed (BS64, LR 3e-4)', '#E4A052', '^'),
    ('R13', 'FD CR, delayed (BS32, LR 1e-4)', '#7B8490', 'D'),
    ('R17', 'FM (BS32, LR 1e-4)', '#4C8997', 'o'),
    ('R20', 'FM+CR, spectral Jacobian (BS32, LR 1e-4)', '#C76870', 'o'),
]


def ema(values, alpha):
    """EMA with previous-value weight alpha, initialized at the first value."""
    if not 0 <= alpha < 1:
        raise ValueError('EMA alpha must satisfy 0 <= alpha < 1.')
    values = np.asarray(values, dtype=float)
    smoothed = values.copy()
    for i in range(1, len(values)):
        smoothed[i] = alpha * smoothed[i-1] + (1-alpha) * values[i]
    return smoothed


def ema_suffix(alpha):
    return '_ema' + f'{alpha:.6f}'.rstrip('0').replace('.', '') if alpha else ''


def ema_note(fig, alpha):
    if alpha:
        fig.text(.5, .05, f'EMA smoothing: s[e] = {alpha:g} s[e-1] + {1-alpha:g} x[e]; s[1] = x[1].',
                 ha='center', fontsize=9, color='#444444')


def smooth_values(values, alpha=0, sigma=0):
    if sigma:
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(np.asarray(values, dtype=float), sigma=sigma,
                                 mode='reflect', truncate=4.0)
    return ema(values, alpha)


def smooth_suffix(alpha=0, sigma=0):
    return '_gaussian_sigma' + f'{sigma:g}' if sigma else ema_suffix(alpha)


def smooth_note(fig, alpha=0, sigma=0):
    if sigma:
        fig.text(.5, .05, f'Gaussian smoothing after normalization: sigma = {sigma:g} epochs; reflect boundaries; endpoints change.',
                 ha='center', fontsize=8.5, color='#444444')
    else:
        ema_note(fig, alpha)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--normalize', action='store_true',
                        help='Divide each run by its own first completed epoch mean.')
    parser.add_argument('--smooth-alpha', type=float, default=0,
                        help='EMA previous-value weight, in [0, 1); default 0 disables smoothing.')
    parser.add_argument('--gaussian-sigma', type=float, default=0,
                        help='Gaussian sigma in epoch points, reflect boundaries; mutually exclusive with EMA.')
    args = parser.parse_args()
    if not 0 <= args.smooth_alpha < 1:
        parser.error('--smooth-alpha must be in [0, 1).')
    if not np.isfinite(args.gaussian_sigma) or args.gaussian_sigma < 0:
        parser.error('--gaussian-sigma must be finite and nonnegative.')
    if args.smooth_alpha and args.gaussian_sigma:
        parser.error('Choose EMA or Gaussian smoothing, not both.')
    with (OUT / 'epoch_loss_summary.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'axes.labelsize': 13, 'xtick.color': 'black',
                         'ytick.color': 'black', 'axes.labelcolor': 'black',
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.subplots_adjust(left=.10, right=.98, top=.96, bottom=.31)
    exported = []
    for run, label, color, marker in SETTINGS:
        data = sorted((r for r in rows if r['run'] == run and 1 <= int(r['epoch']) <= 10),
                      key=lambda r: int(r['epoch']))
        epochs = [int(r['epoch']) for r in data]
        assert epochs == list(range(1, 11)), (run, epochs)
        values = [float(r['total_loss']) for r in data]
        assert np.isfinite(values).all()
        if args.normalize and values[0] <= 0:
            raise ValueError('Normalization requires a positive first epoch loss: ' + run)
        plotted = np.array(values)/values[0] if args.normalize else values
        if args.normalize:
            assert plotted[0] == 1.0
        smoothed = smooth_values(plotted, args.smooth_alpha, args.gaussian_sigma)
        ax.plot(epochs, smoothed, color=color, marker=marker, markersize=5,
                linewidth=2.2, label=label)
        for e, v, p, s in zip(epochs, values, plotted, smoothed):
            record = dict(run=run, setting=label, epoch=e, total_loss=v)
            if args.normalize:
                record.update(first_epoch_total_loss=values[0], normalized_loss=float(p))
            if args.smooth_alpha:
                record.update(EMA_alpha=args.smooth_alpha, smoothed_display_value=float(s))
            if args.gaussian_sigma:
                record.update(Gaussian_sigma=args.gaussian_sigma, boundary_mode='reflect',
                              smoothed_display_value=float(s))
            exported.append(record)
    ax.set(xlabel='Epoch', ylabel=r'Normalized Training Loss ($L_e / L_1$)' if args.normalize
           else 'Total Training Loss', xlim=(.8, 10.2), ylim=(0, 1.05) if args.normalize else (0, .30))
    ax.set_xticks(range(1, 11))
    ax.set_yticks(np.arange(0, 1.01, .2) if args.normalize else np.arange(0, .301, .05))
    ax.grid(color='#E6E6E6', linewidth=.9)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
        spine.set_color('black')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.17), ncol=2,
              frameon=False, fontsize=10, columnspacing=1.6, handlelength=2.6)
    note = ('Each curve is divided by its first completed epoch mean; normalization does not remove setting differences.'
            if args.normalize else 'Original epoch means; objectives, data and initialization differ across settings.')
    fig.text(.5, .025, note, ha='center', fontsize=8.5, color='#444444')
    smooth_note(fig, args.smooth_alpha, args.gaussian_sigma)
    stem = 'six_methods_' + ('normalized' if args.normalize else 'total') + '_training_loss_first10' + smooth_suffix(args.smooth_alpha, args.gaussian_sigma)
    for suffix in ['png', 'pdf', 'svg']:
        path = OUT / (stem + '.' + suffix)
        fig.savefig(path, dpi=600, facecolor='white')
        print(path)
    plt.close(fig)
    with (OUT / (stem + '.csv')).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)


if __name__ == '__main__':
    main()
