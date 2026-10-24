"""Plot all DS-positive paired repeats as an explicitly selected case collection."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--error-bars', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = root / 'outputs/analysis/robustness_complete_summary_20261019'
    source = output / 'positive_DS_complete_summary.csv'
    rows = list(csv.DictReader(source.open()))
    assert len(rows) == 22
    assert all(float(r['DS_CR_minus_FM']) > 0 for r in rows)
    if args.error_bars:
        paired_routes = defaultdict(dict)
        for record in csv.DictReader((output / 'route_metrics.csv').open()):
            key = (record['campaign'], record['phase'], record['condition'], record['route_id'])
            paired_routes[key][record['model']] = record
        rng = np.random.default_rng(20261019)
        for row in rows:
            prefix = (row['campaign'], row['phase'], row['condition'])
            scenarios = defaultdict(list)
            for key, pair in paired_routes.items():
                if key[:3] != prefix:
                    continue
                a, b = pair['FM'], pair['FM_CR']
                assert a['scenario'] == b['scenario']
                scenarios[a['scenario']].append([
                    float(b['DS']) - float(a['DS']),
                    100 * (int(b['SR']) - int(a['SR']))])
            assert len(scenarios) == 44
            values = np.array([np.mean(v, axis=0) for v in scenarios.values()])
            assert np.allclose(values.mean(axis=0),
                               [float(row['DS_CR_minus_FM']), float(row['SR_CR_minus_FM_pp'])], atol=1e-4)
            boot = values[rng.integers(0, 44, (10000, 44))].mean(axis=1)
            for i, metric in enumerate(['DS', 'SR']):
                row[metric + '_CI_low'], row[metric + '_CI_high'] = np.quantile(boot[:, i], [.025, .975])
        with (output / 'positive_DS_cases_with_CI.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    batches = {
        'robustness_campaign_20261014': 'A',
        'robustness_repeat4_seed0_20261015': 'B',
        'robustness_statepulse_5rounds_seed0_20261016': 'C',
        'robustness_positionpulse_5rounds_seed0_20261016': 'D',
        'robustness_inputs5_seed0_20261017': 'E',
        'robustness_inputs5_4groups_seed0_20261018': 'F',
    }
    labels = []
    for row in rows:
        origin = batches[row['campaign']]
        if row['phase'].startswith('repeat'):
            origin += '/' + str(int(row['phase'].replace('repeat', '')))
        labels.append(f'{row["label"]}  [{origin}]')
    y = np.arange(len(rows))
    ds = np.array([float(r['DS_CR_minus_FM']) for r in rows])
    sr = np.array([float(r['SR_CR_minus_FM_pp']) for r in rows])
    sr[np.abs(sr) < 1e-8] = 0
    fig, axes = plt.subplots(1, 2, figsize=(15, 12.2), sharey=True,
                             gridspec_kw={'width_ratios': [1.1, 1]}, layout='constrained')
    colors = []
    for row in rows:
        condition = row['condition']
        if condition == 'clean':
            colors.append('#7d8791')
        elif condition.startswith(('state_', 'pulse_state_')):
            colors.append('#2978b5')
        elif condition.startswith(('feature_', 'pulse_feature_')):
            colors.append('#159487')
        else:
            colors.append('#cb8430')
    axes[0].barh(y, ds, height=.62, color=colors, alpha=.22 if args.error_bars else 1)
    axes[1].barh(y, sr, height=.62,
                 color=['#2978b5' if v > 0 else '#c44e52' if v < 0 else '#b1b7bd' for v in sr],
                 alpha=.22 if args.error_bars else 1)
    if args.error_bars:
        for ax, metric, values, paint in [
            (axes[0], 'DS', ds, colors),
            (axes[1], 'SR', sr, ['#2978b5' if v >= 0 else '#c44e52' for v in sr])]:
            for i, (value, color) in enumerate(zip(values, paint)):
                low, high = rows[i][metric + '_CI_low'], rows[i][metric + '_CI_high']
                ax.errorbar(value, i, xerr=[[max(0, value-low)], [max(0, high-value)]],
                            fmt='o', markersize=4, capsize=4, linewidth=1.3, color=color, zorder=3)
    for i, (a, b) in enumerate(zip(ds, sr)):
        ds_text_x = rows[i]['DS_CI_high'] + .18 if args.error_bars else a + .09
        axes[0].text(ds_text_x, i, f'+{a:.2f}', va='center', fontsize=10)
        if b == 0 and not args.error_bars:
            axes[1].plot(0, i, 'o', color='#8b949e', markersize=4)
        sr_text_x = rows[i]['SR_CI_high'] + .3 if args.error_bars else b + (.16 if b >= 0 else -.16)
        axes[1].text(sr_text_x, i,
                     f'{b:+.2f}' if b else '0.00',
                     va='center', ha='left' if args.error_bars or b >= 0 else 'right', fontsize=10)
    axes[0].set_yticks(y, labels, fontproperties=font, fontsize=10)
    if args.error_bars:
        axes[0].set_xlim(min(r['DS_CI_low'] for r in rows)-.7, max(r['DS_CI_high'] for r in rows)+2.5)
        axes[1].set_xlim(min(r['SR_CI_low'] for r in rows)-1, max(r['SR_CI_high'] for r in rows)+4)
    else:
        axes[0].set_xlim(0, 8.1)
        axes[1].set_xlim(-6.5, 10.6)
    axes[0].set_ylim(len(rows)-.5, -.5)
    for ax in axes:
        ax.axvline(0, color='#555555', linewidth=.9)
        ax.grid(axis='x', alpha=.18)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        for i in range(1, len(rows)):
            if rows[i]['condition'] != rows[i-1]['condition']:
                ax.axhline(i-.5, color='#dce1e5', linewidth=.7)
    axes[0].set_title('A  DS加分（FM+CR − FM）', fontproperties=font, fontsize=13)
    axes[1].set_title('B  对应的成功率变化', fontproperties=font, fontsize=13)
    axes[0].set_xlabel('DS差值', fontproperties=font, fontsize=11)
    axes[1].set_xlabel('SR差值（百分点；FM+CR − FM）', fontproperties=font, fontsize=11)
    fig.suptitle('FM+CR 的 DS 正增益案例：全部 22 个配对轮次' +
                 ('（含场景配对95%区间）' if args.error_bars else ''),
                 fontproperties=font, fontsize=17)
    fig.supxlabel(
        '筛选：单轮 FM+CR DS > FM DS；含无扰动基线1轮、扰动21轮。全部seed0，每轮每模型44条路线。\n'
        'A 原始预实验；B 四条件复测；C 状态脉冲五轮；D 定位脉冲五轮；E 五条件复测；F 最新五条件各四组；/数字为轮次。\n' +
        ('误差线：每轮44场景配对bootstrap 10000次的95%区间，不是重复运行标准差；未校正结果筛选与多重比较。\n'
         if args.error_bars else '') +
        '本图仅展示按DS筛选的改善案例，不代表全部实验均值或统计显著性；SR下降的案例仍保留。',
        fontproperties=font, fontsize=9)
    stem = 'positive_DS_cases_with_errorbars' if args.error_bars else 'positive_DS_cases_overview'
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(output / f'{stem}.{extension}', dpi=220)
    plt.close(fig)
    print(output / f'{stem}.png')


if __name__ == '__main__':
    main()
