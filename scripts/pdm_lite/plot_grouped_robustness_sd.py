"""Five perturbation groups with sample SD across paired repeat-level effects."""
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
GROUPS = [
    ('规划状态扰动', ['state_lateral_0.1', 'state_longitudinal_0.1', 'pulse_state_lateral_0.1']),
    ('BEV 噪声扰动', ['feature_bev_0.03']),
    ('Query 噪声扰动', ['feature_query_0.03', 'pulse_feature_query_0.03']),
    ('定位扰动', ['position_lateral_0.25', 'position_lateral_0.5',
                  'position_longitudinal_0.1', 'position_longitudinal_0.25',
                  'position_longitudinal_0.5', 'pulse_position_lateral_0.25']),
    ('航向偏置扰动', ['heading_1', 'heading_5']),
]
LABELS = {
    'state_lateral_0.1': '持续横向 0.10 m', 'state_longitudinal_0.1': '持续纵向 0.10 m',
    'pulse_state_lateral_0.1': '单次横向脉冲 0.10 m',
    'feature_bev_0.03': '持续 BEV 噪声 0.03', 'feature_query_0.03': '持续 Query 噪声 0.03',
    'pulse_feature_query_0.03': '单次 Query 脉冲 0.03',
    'position_lateral_0.25': '持续横向偏置 0.25 m', 'position_lateral_0.5': '持续横向偏置 0.50 m',
    'position_longitudinal_0.1': '持续纵向偏置 0.10 m',
    'position_longitudinal_0.25': '持续纵向偏置 0.25 m',
    'position_longitudinal_0.5': '持续纵向偏置 0.50 m',
    'pulse_position_lateral_0.25': '单次横向脉冲 0.25 m',
    'heading_1': '持续航向偏置 1°', 'heading_5': '持续航向偏置 5°',
}


def summarize(rows, scope):
    grouped = defaultdict(list)
    for r in rows:
        if r['condition'] != 'clean':
            grouped[r['condition']].append(r)
    summaries = []
    for group, conditions in GROUPS:
        for condition in conditions:
            selected = grouped[condition]
            if not selected:
                continue
            result = dict(scope=scope, group=group, condition=condition, label=LABELS[condition],
                          paired_rounds=len(selected), SD_available=len(selected) >= 2)
            for metric, key in [('DS', 'DS_CR_minus_FM'), ('SR_pp', 'SR_CR_minus_FM_pp')]:
                values = np.array([float(r[key]) for r in selected])
                result[metric + '_mean'] = float(values.mean())
                result[metric + '_sample_SD'] = float(values.std(ddof=1)) if len(values) >= 2 else None
                result[metric + '_round_values'] = '|'.join(str(v) for v in values)
            summaries.append(result)
    return summaries


def plot(rows, scope, title):
    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    heights = [sum(r['group'] == name for r in rows) + .8 for name, _ in GROUPS]
    fig, axes = plt.subplots(5, 2, figsize=(15, 13),
                             gridspec_kw={'height_ratios': heights}, layout='constrained')
    for i, (group, _) in enumerate(GROUPS):
        selected = [r for r in rows if r['group'] == group]
        for j, (metric, xlabel) in enumerate([('DS', 'DS差值：FM+CR − FM'),
                                             ('SR_pp', 'SR差值：FM+CR − FM（百分点）')]):
            ax = axes[i, j]
            for k, r in enumerate(selected):
                mean, sd = r[metric + '_mean'], r[metric + '_sample_SD']
                values = np.array([float(v) for v in r[metric + '_round_values'].split('|')])
                jitter = np.linspace(-.12, .12, len(values)) if len(values) > 1 else np.array([0.])
                ax.scatter(values, k + jitter, marker='x', s=24, color='#8b949e', alpha=.8, zorder=2)
                color = '#2171b5' if mean >= 0 else '#c44e52'
                if sd is not None:
                    ax.errorbar(mean, k, xerr=sd, fmt='o', color=color, capsize=4,
                                markersize=6, linewidth=1.6, zorder=3)
                else:
                    ax.plot(mean, k, 'D', color=color, markersize=6, zorder=3)
            ax.axvline(0, color='#555555', linewidth=.9, linestyle='--')
            ax.set_ylim(len(selected)-.5, -.5)
            ax.set_yticks(range(len(selected)))
            if j == 0:
                ax.set_yticklabels([r['label'] + f'  [n={r["paired_rounds"]}' +
                                     ('，SD不可估计]' if not r['SD_available'] else ']')
                                     for r in selected], fontproperties=font, fontsize=10)
            else:
                ax.set_yticklabels([])
            ax.set_title(group + ('：DS' if j == 0 else '：SR'), fontproperties=font, fontsize=12)
            ax.set_xlabel(xlabel, fontproperties=font, fontsize=10)
            ax.grid(axis='x', alpha=.18)
            ax.margins(x=.15)
            ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(title, fontproperties=font, fontsize=17)
    fig.supxlabel(
        '圆点与横线：同条件配对轮次差值的均值 ± 1个样本标准差（ddof=1）；灰色×：每轮差值。\n'
        '菱形：仅1轮，标准差不可估计，不画误差线。n为配对轮数；每轮每模型44路线，全部seed0。\n'
        '不同方向、幅度和持续／脉冲不混合计算；标准差描述运行间波动，不是置信区间。' +
        ('\n本图仅纳入DS正增益轮次，标准差受筛选影响，不代表全部实验波动。' if scope == 'positive_only' else ''),
        fontproperties=font, fontsize=9)
    stem = 'grouped_repeat_SD_' + scope
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / f'{stem}.{ext}', dpi=220)
    plt.close(fig)


def main():
    all_rows = list(csv.DictReader((OUT / 'paired_rounds.csv').open()))
    positive = [r for r in all_rows if float(r['DS_CR_minus_FM']) > 0]
    results = []
    for scope, source, title in [
        ('positive_only', positive, '五类扰动：DS正增益案例的均值与轮间标准差'),
        ('all_rounds', all_rows, '五类扰动：全部轮次的均值与轮间标准差')]:
        rows = summarize(source, scope)
        results.extend(rows)
        plot(rows, scope, title)
    with (OUT / 'grouped_repeat_SD_statistics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    notes = ['# 五类扰动：配对轮次标准差', '',
             '横线是同条件的配对轮次差值均值±1个样本标准差，不是bootstrap区间或标准误。',
             '标准差公式：s=sqrt(sum((d_r-mean(d))^2)/(n-1))，d_r为某轮44路线平均CR-FM差值。',
             '只有一轮的条件SD留空，不设为0；没有混合不同条件增加样本数。',
             'positive_only沿用DS正增益筛选；all_rounds包含所有完成轮次。clean另存原始表，不属于五类扰动。',
             '全部为seed0重复，SD仅描述当前设置下观察到的运行波动。至少2轮才能估计SD；2轮估计很不稳定。', '',
             '| 范围 | 条件 | 轮数 | DS均值 | DS标准差 |', '|---|---|---:|---:|---:|']
    for r in results:
        sd = f'{r["DS_sample_SD"]:.3f}' if r['SD_available'] else '不可估计'
        notes.append(f'| {r["scope"]} | {r["label"]} | {r["paired_rounds"]} | {r["DS_mean"]:+.3f} | {sd} |')
    (OUT / 'grouped_repeat_SD_README.md').write_text('\n'.join(notes) + '\n')
    for r in results:
        print(r['scope'], r['condition'], r['paired_rounds'], r['DS_mean'], r['DS_sample_SD'])


if __name__ == '__main__':
    main()
