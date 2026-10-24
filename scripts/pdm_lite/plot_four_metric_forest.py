"""English four-metric forest plot of the existing DS-positive pooled case subset."""
import csv
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
from matplotlib.ticker import MultipleLocator
import numpy as np


def main():
    root = Path(__file__).resolve().parents[2]
    out = root / 'outputs/analysis/robustness_complete_summary_20261019'
    read = lambda name: list(csv.DictReader((out / name).open()))
    primary = [r for r in read('merged_categories_SD_statistics.csv') if r['scope'] == 'positive_only']
    extra = {r['group']: r for r in read('grouped_comfort_efficiency_statistics.csv')
             if r['scope'] == 'positive_only'}
    english = ['Accumulated Error', 'BEV Noise Perturbations', 'Query Noise Perturbations',
               'Localization Perturbations', 'Heading Perturbations']
    fontManager.addfont('/usr/share/fonts/truetype/msttcorefonts/Arial.ttf')
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 14,
                         'axes.labelsize': 14, 'xtick.labelsize': 14,
                         'ytick.labelsize': 14, 'axes.unicode_minus': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    specs = [
        ('Driving Score', '#4C8997', -.21, 'DS_mean', 'DS_sample_SD', 'score points'),
        ('Success Rate', '#C76870', -.07, 'SR_pp_mean', 'SR_pp_sample_SD', 'percentage points'),
        ('Comfort', '#B7D3E0', .07, 'Comfort_CR_minus_FM_mean', 'Comfort_CR_minus_FM_SD', 'percentage points'),
        ('Efficiency', '#E9B389', .21, 'Efficiency_CR_minus_FM_mean', 'Efficiency_CR_minus_FM_SD', 'benchmark percentage-score points'),
    ]
    fig, ax = plt.subplots(figsize=(11, 5.4))
    fig.subplots_adjust(left=.27, right=.98, bottom=.17, top=.97)
    y = np.arange(5)
    export = []
    for metric, color, offset, mean_key, sd_key, unit in specs:
        records = primary if metric in ['Driving Score', 'Success Rate'] else [extra[r['group']] for r in primary]
        assert all(a['paired_rounds'] == b['paired_rounds'] for a, b in zip(primary, records))
        mean = np.array([float(r[mean_key]) for r in records])
        sd = np.array([float(r[sd_key]) for r in records])
        sd[np.abs(sd) < 1e-12] = 0
        ax.errorbar(mean, y+offset, xerr=sd, fmt='o', markersize=7,
                    markerfacecolor='none', markeredgecolor=color, markeredgewidth=1.8,
                    color=color, linewidth=1.6, capsize=0, label=metric, zorder=4)
        for i, r in enumerate(records):
            export.append(dict(category=english[i], metric=metric,
                               paired_rounds=int(r['paired_rounds']),
                               mean_CR_minus_FM=mean[i], sample_SD=sd[i], unit=unit,
                               scope='DS-positive paired rounds only'))
    wrapped_labels = [textwrap.fill(label, width=20, break_long_words=False,
                                    break_on_hyphens=False) for label in english]
    ax.set_yticks(y, wrapped_labels)
    for label in ax.get_yticklabels():
        label.set_color('#000000')
        label.set_multialignment('right')
    ax.set_ylim(4.5, -.5)
    ax.set_xlim(-16, 15)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(2.5))
    ax.grid(which='major', color='#E7E7E7', linewidth=1.15)
    ax.grid(which='minor', axis='x', color='#F0F0F0', linewidth=.9)
    ax.axvline(0, color='#333333', linestyle='--', linewidth=1.1, zorder=3)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color('#333333')
        spine.set_linewidth(.9)
    ax.tick_params(axis='both', colors='#000000', length=4, width=.8)
    ax.tick_params(axis='x', which='minor', length=0)
    ax.set_xlabel('Performance Difference', labelpad=8, color='#000000')
    ax.legend(loc='lower right', frameon=False, fontsize=12,
              handlelength=1.7, labelspacing=.6, borderaxespad=.5)
    stem = 'four_metric_forest_english_black_wrapped_600dpi'
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(out / f'{stem}.{extension}', dpi=600, facecolor='white')
    plt.close(fig)
    with (out / f'{stem}_data.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(export[0]))
        writer.writeheader()
        writer.writerows(export)
    (out / f'{stem}_README.txt').write_text(
        'Existing selected DS-positive subset: 21 perturbation paired rounds; clean excluded.\n'
        'All differences are FM+CR minus FM. Error lines: +/-1 sample SD, ddof=1.\n'
        'Heterogeneous conditions are pooled within categories; each round has equal weight.\n'
        'SD includes condition variation and repeated-run variation; not a confidence interval.\n'
        'Driving Score uses score points; Success Rate and Comfort differences use percentage points.\n'
        'Efficiency is the existing local Bench2Drive percentage-score metric; values can exceed 100.\n'
        'No normalization or rescaling was applied to any of the four metrics.\n'
        'English category names follow the user request, including Accumulated Error.\n'
        'Arial: axis labels and ticks 14pt, legend 12pt, matching the preceding style.\n'
        'Tick labels pure black; English category labels wrapped at word boundaries (20 characters).\n'
        'Accumulated Error Efficiency remains the observed -3.5337494668 +/- 11.0812449440; no manual adjustment.\n'
        '600 dpi PNG, 6600 x 3240 pixels. Selected cases do not represent all experiments.\n')
    print(out / f'{stem}.png')


if __name__ == '__main__':
    main()
