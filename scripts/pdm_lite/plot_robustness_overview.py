"""Paper overview using every completed repeat, including unfavorable outcomes."""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/analysis/robustness_complete_summary_20261019'


def main():
    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', size=10)
    rows = list(csv.DictReader((OUT / 'condition_summary.csv').open()))
    rows = [r for r in rows if r['condition'] != 'clean']
    pairs = defaultdict(dict)
    for r in csv.DictReader((OUT / 'route_metrics.csv').open()):
        pairs[r['campaign'], r['phase'], r['condition'], r['route_id']][r['model']] = r
    scenario_sr = defaultdict(lambda: defaultdict(list))
    for (_, _, condition, _), pair in pairs.items():
        if condition == 'clean':
            continue
        a, b = pair['FM'], pair['FM_CR']
        scenario_sr[condition][a['scenario']].append(100 * (int(b['SR']) - int(a['SR'])))
    rng = np.random.default_rng(20261019)
    for r in rows:
        values = np.array([np.mean(v) for v in scenario_sr[r['condition']].values()])
        assert len(values) == 44
        boot = values[rng.integers(0, 44, (10000, 44))].mean(axis=1)
        r['SR_scenario_bootstrap_low'], r['SR_scenario_bootstrap_high'] = np.quantile(boot, [.025, .975])
    with (OUT / 'figure_overview_data.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(15, 8.8), sharey=True,
                             gridspec_kw={'width_ratios': [1.15, 1]}, layout='constrained')
    y = np.arange(len(rows))
    labels = [f'{r["label"]}  [n={r["paired_rounds"]}; DS胜出{r["CR_DS_wins"]}/{r["paired_rounds"]}]'
              for r in rows]
    for ax, metric, lowkey, highkey, title, xlabel in [
        (axes[0], 'DS_CR_minus_FM', 'absolute_DS_gap_scenario_bootstrap_low',
         'absolute_DS_gap_scenario_bootstrap_high', 'A  规划与驾驶综合成绩', 'DS差：FM+CR − FM'),
        (axes[1], 'SR_CR_minus_FM_pp', 'SR_scenario_bootstrap_low',
         'SR_scenario_bootstrap_high', 'B  路线成功率', 'SR差：FM+CR − FM（百分点）')]:
        ax.axvline(0, linestyle='--', color='#555555', linewidth=1)
        for i, r in enumerate(rows):
            value, low, high = (float(r[k]) for k in [metric, lowkey, highkey])
            color = '#2171b5' if value >= 0 else '#c44e52'
            ax.errorbar(value, i, xerr=[[max(0, value-low)], [max(0, high-value)]],
                        fmt='o', color=color, markersize=5, capsize=3, linewidth=1.4)
        for top, bottom in [(-.5, 2.5), (2.5, 5.5)]:
            ax.axhspan(top, bottom, color='#f2f5f8' if top < 0 else '#fff8ed', zorder=-1)
        for line in [2.5, 5.5]:
            ax.axhline(line, color='#cccccc', linewidth=.8)
        ax.set_title(title, fontproperties=font, fontsize=13, pad=12)
        ax.set_xlabel(xlabel, fontproperties=font, fontsize=11)
        ax.grid(axis='x', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_yticks(y, labels, fontproperties=font)
    axes[0].set_ylim(len(rows)-.5, -.5)
    fig.suptitle('FM+CR闭环扰动实验：全部已完成配对轮次', fontproperties=font, fontsize=16)
    fig.supxlabel('点：全部重复均值；线：场景配对bootstrap 95%区间；右侧为CR更好。'
                  '全部seed0，每轮44路线；n为配对轮数。\n'
                  '比较扰动下绝对成绩，未扣除clean基线；区间不代表跨训练种子不确定性。',
                  fontproperties=font, fontsize=10)
    for ext in ['png', 'pdf']:
        fig.savefig(OUT / f'paper_robustness_overview.{ext}', dpi=220)
    plt.close(fig)
    print(OUT / 'paper_robustness_overview.png')


if __name__ == '__main__':
    main()
