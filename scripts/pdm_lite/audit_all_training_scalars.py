"""Inventory and plot every TensorBoard scalar of the evaluated FM/FM+CR runs."""
import csv
import gzip
import html
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from scripts.pdm_lite.audit_training_curves_fm_cr import ROOT, RUNS

OUT = ROOT / 'outputs/analysis/fm_cr_all_training_scalars_20261019'


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scalars, inventories, epochs, blocks = {}, [], [], []
    for model, folder in RUNS.items():
        print('Reading all scalar records:', model, flush=True)
        event = EventAccumulator(str(folder), size_guidance={'scalars': 0})
        event.Reload()
        with gzip.open(OUT / ('FM_CR_raw_scalars.csv.gz' if model == 'FM+CR' else 'FM_raw_scalars.csv.gz'),
                       'wt', newline='', compresslevel=6) as raw:
            writer = csv.writer(raw)
            writer.writerow(['model', 'tag', 'step', 'value', 'wall_time'])
            for tag in event.Tags()['scalars']:
                series = event.Scalars(tag)
                steps = np.array([r.step for r in series])
                values = np.array([r.value for r in series])
                writer.writerows((model, tag, r.step, r.value, r.wall_time) for r in series)
                finite = np.isfinite(values)
                assert finite.all(), (model, tag)
                epoch_level = len(series) == 10 and np.array_equal(steps, np.arange(10))
                all_zero = bool(np.all(values == 0))
                kind = 'loss' if 'loss' in tag.lower() else 'diagnostic'
                inventory = dict(model=model, tag=tag, kind=kind, samples=len(series),
                                 level='epoch' if epoch_level else 'step',
                                 first_step=int(steps[0]), last_step=int(steps[-1]),
                                 first_value=float(values[0]), last_value=float(values[-1]),
                                 minimum=float(values.min()), maximum=float(values.max()),
                                 mean=float(values.mean()), all_zero=all_zero,
                                 nonfinite_count=int((~finite).sum()))
                inventories.append(inventory)
                scalars[model, tag] = dict(steps=steps, values=values, epoch_level=epoch_level,
                                         all_zero=all_zero, kind=kind)
                if epoch_level:
                    for step, value in zip(steps, values):
                        epochs.append(dict(model=model, tag=tag, epoch=int(step)+1,
                                           count=1, mean=float(value), SD=None,
                                           origin='logged_epoch_mean'))
                else:
                    for width, destination, key in [(9227, epochs, 'epoch'), (500, blocks, 'block')]:
                        bins = (steps-1)//width
                        for index in np.unique(bins):
                            mask = bins == index
                            a = values[mask]
                            if key == 'epoch':
                                destination.append(dict(model=model, tag=tag, epoch=int(index)+1,
                                                        count=len(a), mean=float(a.mean()),
                                                        SD=float(a.std(ddof=1)) if len(a)>1 else None,
                                                        origin='recorded_batch_mean'))
                            else:
                                destination.append(dict(model=model, tag=tag, block=int(index),
                                                        mean_epoch=float(steps[mask].mean()/9227),
                                                        count=len(a), mean=float(a.mean()),
                                                        SD=float(a.std(ddof=1)) if len(a)>1 else None))
        print('Exported', model, len(event.Tags()['scalars']), 'tags', flush=True)
    write_csv('scalar_inventory.csv', inventories)
    write_csv('all_epoch_metrics.csv', epochs)
    write_csv('all_500step_blocks.csv', blocks)
    aliases = []
    for model in RUNS:
        keys = [tag for run, tag in scalars if run == model]
        for i, left in enumerate(keys):
            for right in keys[i+1:]:
                a, b = scalars[model, left], scalars[model, right]
                if (not a['all_zero'] and not b['all_zero']
                        and np.array_equal(a['steps'], b['steps'])
                        and np.array_equal(a['values'], b['values'])):
                    aliases.append(dict(model=model, left=left, right=right, relation='exactly_equal_logged_series'))
    write_csv('identical_nonzero_series.csv', aliases)
    tags = sorted({tag for _, tag in scalars})
    active_loss = [tag for tag in tags if 'loss' in tag and
                   any(not scalars[model, tag]['all_zero'] for model in RUNS if (model, tag) in scalars)]
    zero_loss = [tag for tag in tags if 'loss' in tag and tag not in active_loss]
    diagnostics = [tag for tag in tags if 'loss' not in tag]
    block_lookup = defaultdict(list)
    for r in blocks:
        block_lookup[r['model'], r['tag']].append(r)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42})
    pages, drawn = [], []
    with PdfPages(OUT / 'all_training_curves.pdf') as pdf:
        for kind, selected in [('active_losses', active_loss), ('zero_losses', zero_loss),
                               ('diagnostics', diagnostics)]:
            for offset in range(0, len(selected), 9):
                chunk = selected[offset:offset+9]
                fig, axes = plt.subplots(3, 3, figsize=(14, 10), layout='constrained')
                for ax, tag in zip(axes.flat, chunk):
                    drawn.append(tag)
                    for model, color in [('FM', '#4C8997'), ('FM+CR', '#C76870')]:
                        if (model, tag) not in scalars:
                            continue
                        item = scalars[model, tag]
                        if item['epoch_level']:
                            x, y = item['steps']+1, item['values']
                            ax.plot(x, y, '-o', markersize=3, color=color, label=model, linewidth=1.5)
                        else:
                            records = block_lookup[model, tag]
                            ax.plot([r['mean_epoch'] for r in records], [r['mean'] for r in records],
                                    color=color, label=model, linewidth=1.3)
                        if kind == 'zero_losses':
                            ax.set_ylim(-.05, .05)
                    ax.set(title=tag, xlabel='Training epoch', xlim=(0, 10))
                    ax.axvline(2, color='#aaaaaa', linestyle='--', linewidth=.8)
                    ax.axvspan(2, 3, color='#f1e5d6', alpha=.35, zorder=-1)
                    ax.grid(alpha=.2)
                    ax.legend(frameon=False, fontsize=8)
                    if kind == 'zero_losses':
                        ax.text(.5, .75, 'All logged values are zero', transform=ax.transAxes,
                                ha='center', color='#777777', fontsize=9)
                for ax in list(axes.flat)[len(chunk):]:
                    ax.set_visible(False)
                fig.suptitle(kind.replace('_', ' ').title() + f' | page {offset//9+1}', fontsize=15)
                fig.supxlabel('Epoch series: original logged values; step series: 500-step block means. '
                              'Dashed line / shading: CR start / ramp. Training seed0 only.\n'
                              'Zero loss entries are inactive task placeholders in this FMSR training setup; '
                              'missing CR observations are not filled with zeros.', fontsize=9)
                stem = f'{kind}_{offset//9+1:02d}'
                fig.savefig(OUT / f'{stem}.png', dpi=400)
                pdf.savefig(fig)
                plt.close(fig)
                pages.append(dict(group=kind, filename=stem+'.png', tags=chunk))
    assert set(drawn) == set(tags) and len(drawn) == len(tags)
    audit = dict(runs={model:dict(tags=sum(r['model']==model for r in inventories),
                                loss_tags=sum(r['model']==model and r['kind']=='loss' for r in inventories))
                       for model in RUNS}, unique_tags=len(tags),
                 active_loss_tags=active_loss, zero_loss_tags=zero_loss,
                 diagnostic_tags=diagnostics, gallery_pages=pages,
                 no_validation_tags=not any(t.startswith(('val_', 'val/', 'validation/', 'test/')) for t in tags),
                 all_records_finite=True, observed_identical_series=aliases,
                 source_directories={k:str(v) for k,v in RUNS.items()})
    (OUT / 'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    doc = ['# FM / FM+CR全部训练曲线检查', '',
           f'共{len(tags)}种不同标量：{len(active_loss)}种非全零loss标签、{len(zero_loss)}种全零loss标签、'
           f'{len(diagnostics)}种诊断标签。全部观测值有限；未发现独立验证指标。', '',
           '同一loss可能同时记录逐步与逐epoch版本，不是相互独立的训练目标。',
           'FM+CR的回归loss包含CR项，不能与FM回归loss直接作同目标高低比较。',
           '本轮训练代码仅对loss_fmsr_reg和loss_fmsr_cls设权重1；其余任务loss为0权重，'
           '因此全零曲线不是这些感知任务已完美拟合。',
           '代码依据：fmsr_pdm_lite/vendor/garage/train.py的use_fmsr分支；'
           'model.py返回回归/分类损失；logger对预留项记录为0。',
           '图内逐步序列按500步分箱平均，逐epoch序列使用原始10点；原始所有标量另存csv.gz。',
           '统计中的SD是批次波动，不是多训练种子的误差条。', '',
           '## 非全零loss', ''] + ['- '+tag for tag in active_loss]
    doc += ['', '## 全零loss', ''] + ['- '+tag for tag in zero_loss]
    doc += ['', '## 所有loss最终epoch均值', '',
            '| 标签 | FM | FM+CR |', '|---|---:|---:|']
    for tag in active_loss:
        vals=[]
        for model in RUNS:
            a=[r for r in epochs if r['model']==model and r['tag']==tag and r['epoch']==10]
            vals.append(f'{a[0]["mean"]:.7f}' if a else '未记录／不适用')
        doc.append(f'| {tag} | {vals[0]} | {vals[1]} |')
    doc += ['', '## 诊断图集', '']
    for page in pages:
        doc += [f'### {page["filename"]}', '', f'![{page["group"]}]({page["filename"]})', '']
    (OUT / 'REPORT.md').write_text('\n'.join(doc)+'\n')
    body = ['<!doctype html><meta charset="utf-8"><title>All training curves</title>',
            '<style>body{font:16px sans-serif;max-width:1500px;margin:30px auto}img{width:100%}p{line-height:1.6}</style>',
            '<h1>FM / FM+CR：全部训练曲线</h1>',
            '<p>所有标签均已绘出。逐步序列为500步分箱平均；全零loss为本次训练未启用的任务占位。'
            '单训练seed，无独立验证集曲线。</p>']
    for page in pages:
        body.extend([f'<h2>{html.escape(page["filename"])}</h2>',
                     '<p>'+html.escape(', '.join(page['tags']))+'</p>',
                     f'<img src="{page["filename"]}" alt="{page["group"]}">'])
    (OUT / 'index.html').write_text('\n'.join(body))
    print(json.dumps({k:v for k,v in audit.items() if k not in ['gallery_pages', 'observed_identical_series']}, indent=2), flush=True)
    print('Saved', OUT, flush=True)


if __name__ == '__main__':
    main()
