"""Pool heterogeneous perturbation conditions within five explicitly labeled categories."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

from scripts.pdm_lite.plot_grouped_robustness_sd import GROUPS

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/analysis/robustness_complete_summary_20261019'


def make_plot(source, scope):
    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    statistics = []
    selected_rows = []
    for group, conditions in GROUPS:
        rows = [r for r in source if r['condition'] in conditions]
        assert len(rows) >= 2
        r = dict(scope=scope, group=group, paired_rounds=len(rows),
                 distinct_conditions=len({v['condition'] for v in rows}),
                 conditions='|'.join(dict.fromkeys(v['condition'] for v in rows)))
        for metric, key in [('DS', 'DS_CR_minus_FM'), ('SR_pp', 'SR_CR_minus_FM_pp')]:
            values = np.array([float(v[key]) for v in rows])
            r[metric + '_mean'] = float(values.mean())
            r[metric + '_sample_SD'] = float(values.std(ddof=1))
            r[metric + '_values'] = '|'.join(str(v) for v in values)
        statistics.append(r)
        selected_rows.extend(dict(group=group, **v) for v in rows)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.4), sharey=True, layout='constrained')
    colors = ['#2978b5', '#159487', '#6c75b8', '#cb8430', '#b96d89']
    for ax, metric, title, xlabel in [
        (axes[0], 'DS', 'A  DS差异：合并均值 ± 标准差', 'DS差值（FM+CR − FM）'),
        (axes[1], 'SR_pp', 'B  成功率差异：合并均值 ± 标准差', 'SR差值（百分点；FM+CR − FM）')]:
        for i, (r, color) in enumerate(zip(statistics, colors)):
            mean, sd = r[metric + '_mean'], r[metric + '_sample_SD']
            values = np.array([float(v) for v in r[metric + '_values'].split('|')])
            jitter = np.linspace(-.13, .13, len(values))
            ax.barh(i, mean, height=.45, color=color, alpha=.15)
            ax.scatter(values, i+jitter, marker='x', s=26, color='#8b949e', alpha=.75, zorder=2)
            ax.errorbar(mean, i, xerr=sd, fmt='o', color=color, capsize=5,
                        markersize=7, linewidth=2, zorder=3)
            ax.text(.98, i, f'{mean:+.2f} ± {sd:.2f}', transform=ax.get_yaxis_transform(),
                    ha='right', va='center', fontsize=10,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.8, pad=1))
        ax.set_title(title, fontproperties=font, fontsize=13)
        ax.set_xlabel(xlabel, fontproperties=font, fontsize=11)
        ax.axvline(0, color='#555555', linestyle='--', linewidth=1)
        ax.set_ylim(4.5, -.5)
        ax.margins(x=.23)
        ax.grid(axis='x', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_yticks(range(5), [r['group'] + f'  [n={r["paired_rounds"]}；{r["distinct_conditions"]}种条件]'
                                 for r in statistics], fontproperties=font, fontsize=11)
    fig.suptitle('五类扰动合并统计：' + ('DS正增益案例' if scope == 'positive_only' else '全部已完成轮次'),
                 fontproperties=font, fontsize=17)
    note = ('仅纳入DS正增益轮次，共21轮；该筛选子集不代表全部实验。' if scope == 'positive_only'
            else '纳入全部40个扰动配对轮次，不按增益筛选。')
    fig.supxlabel(
        '圆点及误差线：类别内配对轮次差值的均值 ± 1个样本标准差（ddof=1）；灰色×：每轮差值。\n'
        '类别内混合幅度、方向、持续／脉冲；每轮等权，每轮每模型44条路线，全部seed0；无扰动基线不纳入。\n'
        '标准差包含条件间差异与运行波动，不是置信区间，也不是单个固定条件的重复稳定性。\n' + note,
        fontproperties=font, fontsize=9)
    stem = 'merged_categories_SD_' + scope
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / f'{stem}.{ext}', dpi=220)
    plt.close(fig)
    with (OUT / f'{stem}_contributing_rounds.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(selected_rows[0]))
        writer.writeheader()
        writer.writerows(selected_rows)
    return statistics


def main():
    rows = list(csv.DictReader((OUT / 'paired_rounds.csv').open()))
    positive = [r for r in rows if float(r['DS_CR_minus_FM']) > 0]
    results = make_plot(positive, 'positive_only') + make_plot(rows, 'all_rounds')
    with (OUT / 'merged_categories_SD_statistics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    for r in results:
        print(r['scope'], r['group'], r['paired_rounds'],
              f'DS={r["DS_mean"]:+.3f}±{r["DS_sample_SD"]:.3f}',
              f'SR={r["SR_pp_mean"]:+.3f}±{r["SR_pp_sample_SD"]:.3f}')


if __name__ == '__main__':
    main()
