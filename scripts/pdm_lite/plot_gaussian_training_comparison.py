"""Compare unsmoothed and Gaussian-smoothed normalized loss/ADE/FDE.

Run: python scripts/pdm_lite/plot_gaussian_training_comparison.py
Sigma is measured in epoch data points; reflect boundaries, truncate=4.
Normalization precedes filtering. Do not re-anchor filtered endpoints to 1.
"""
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plot_six_training_losses import OUT, SETTINGS, smooth_values


def main():
    with (OUT / 'epoch_loss_summary.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.labelsize': 12, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    fig.subplots_adjust(left=.065, right=.985, bottom=.19, top=.955, hspace=.38, wspace=.23)
    exported = []
    for ri, sigma in enumerate([0, 3, 5]):
        for ci, (metric, key) in enumerate([('Loss', 'total_loss'), ('ADE', 'train_ADE'), ('FDE', 'train_FDE')]):
            ax = axes[ri, ci]
            for run, label, color, marker in SETTINGS:
                data = sorted((r for r in rows if r['run'] == run and 1 <= int(r['epoch']) <= 10),
                              key=lambda r: int(r['epoch']))
                assert [int(r['epoch']) for r in data] == list(range(1, 11))
                original = np.array([float(r[key]) for r in data])
                normalized = original/original[0]
                values = smooth_values(normalized, sigma=sigma)
                assert np.isfinite(values).all()
                # Reflect Gaussian filtering preserves the sum of this sequence.
                assert np.isclose(values.sum(), normalized.sum(), rtol=1e-12)
                ax.plot(range(1, 11), values, color=color, marker=marker, ms=3.5, lw=1.8, label=label)
                exported.extend(dict(run=run, metric=metric, sigma=sigma, epoch=i+1,
                                     normalized_original=float(normalized[i]), displayed_value=float(v),
                                     boundary_mode='reflect',truncate=4)
                                for i,v in enumerate(values))
            name = 'No smoothing' if sigma == 0 else rf'Gaussian $\sigma={sigma}$'
            ax.set_title(metric + ' | ' + name, fontsize=12)
            ax.set(xlim=(.8, 10.2), ylim=(0, 1.05), xlabel='Epoch', ylabel='Normalized ' + metric)
            ax.set_xticks([1, 2, 4, 6, 8, 10])
            ax.set_yticks(np.arange(0, 1.01, .2))
            ax.grid(color='#E6E6E6', linewidth=.8)
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_color('black')
                spine.set_linewidth(1)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.52, .045), ncol=2,
               frameon=False, fontsize=10, columnspacing=1.8)
    fig.text(.5, .02, 'Normalize by epoch 1, then filter (reflect boundaries). Gaussian filtering changes endpoints; settings differ across runs.',
             ha='center', fontsize=9, color='#444444')
    stem = 'six_methods_normalized_gaussian_comparison'
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / (stem + '.' + extension), dpi=600, facecolor='white')
    plt.close(fig)
    with (OUT / (stem + '.csv')).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
    print('Saved Gaussian comparison:', OUT / (stem + '.png'))


if __name__ == '__main__':
    main()
