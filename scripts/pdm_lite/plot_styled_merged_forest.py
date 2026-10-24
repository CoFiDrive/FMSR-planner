"""Compact reference-style overlaid forest plot; selected cases and pooled sample SD."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager, FontProperties
from matplotlib.ticker import MultipleLocator
import numpy as np


def main():
    root = Path(__file__).resolve().parents[2]
    out = root / 'outputs/analysis/robustness_complete_summary_20261019'
    rows = [r for r in csv.DictReader((out / 'merged_categories_SD_statistics.csv').open())
            if r['scope'] == 'positive_only']
    assert len(rows) == 5
    fontManager.addfont('/usr/share/fonts/truetype/msttcorefonts/Arial.ttf')
    fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({
        'font.family': ['Arial'], 'font.size': 14,
        'axes.labelsize': 14, 'xtick.labelsize': 14, 'ytick.labelsize': 14,
        'axes.unicode_minus': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
    })
    ds_color, sr_color = '#4C8997', '#C76870'
    fig, ax = plt.subplots(figsize=(7.8, 4.1))
    fig.subplots_adjust(left=.30, right=.98, bottom=.20, top=.96)
    y = np.arange(5)
    for metric, offset, color, label in [
        ('DS', -.075, ds_color, 'DS'),
        ('SR_pp', .075, sr_color, 'SR (pp)')]:
        means = np.array([float(r[metric + '_mean']) for r in rows])
        sd = np.array([float(r[metric + '_sample_SD']) for r in rows])
        sd[np.abs(sd) < 1e-12] = 0
        ax.errorbar(means, y+offset, xerr=sd, fmt='o', markersize=7,
                    markerfacecolor='none', markeredgewidth=1.8, markeredgecolor=color,
                    color=color, linewidth=1.6, capsize=0, label=label, zorder=4)
    chinese = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', size=14)
    ax.set_yticks(y, [r['group'] for r in rows], fontproperties=chinese)
    for label in ax.get_yticklabels():
        label.set_color('#555555')
    ax.set_ylim(4.5, -.5)
    ax.set_xlim(-4, 10)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(which='major', color='#E7E7E7', linewidth=1.15)
    ax.grid(which='minor', axis='x', color='#F0F0F0', linewidth=.9)
    ax.axvline(0, color='#333333', linestyle='--', linewidth=1.1, zorder=3)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color('#333333')
        spine.set_linewidth(.9)
    ax.tick_params(axis='both', colors='#555555', length=4, width=.8)
    ax.tick_params(axis='x', which='minor', length=0)
    ax.set_xlabel('FM+CR − FM', labelpad=7)
    ax.legend(loc='lower right', frameon=False, fontsize=12, handlelength=1.7,
              labelspacing=.5, borderaxespad=.55)
    stem = 'merged_positive_forest_reference_style_600dpi'
    fig.savefig(out / (stem + '.png'), dpi=600, facecolor='white')
    fig.savefig(out / (stem + '.pdf'), facecolor='white')
    fig.savefig(out / (stem + '.svg'), facecolor='white')
    plt.close(fig)
    (out / (stem + '_README.txt')).write_text(
        'Source: merged_categories_SD_statistics.csv; scope=positive_only.\n'
        'Five pooled perturbation categories, 21 DS-positive paired rounds; clean excluded.\n'
        'Points: mean CR-FM; horizontal lines: +/-1 sample SD (ddof=1).\n'
        'DS uses score points; SR uses percentage points (pp), not relative percent.\n'
        'Within-category condition differences and run variability are pooled.\n'
        'Selected positive cases do not represent all runs or statistical significance.\n'
        '600 dpi; 4680 x 2460 pixels. Arial Latin, Noto Sans CJK SC Chinese fallback; 14pt axes.\n')
    print(out / (stem + '.png'))


if __name__ == '__main__':
    main()
