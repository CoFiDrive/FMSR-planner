"""Audit local training settings and epoch losses, without training or editing logs.

Run from repository root: python scripts/pdm_lite/audit_local_training.py
Event records are read with protobuf from read-only mmap; framing and lengths are
checked. TFRecord CRCs are not checked by this fast reader. The final audit also
cross-checks a small file with the standard TensorBoard reader.
"""
import argparse
import csv
import hashlib
import json
import math
import mmap
import html
from pathlib import Path
import struct
import time
from collections import defaultdict

from tensorboard.compat.proto import event_pb2

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/local_training'
OUT = ROOT / 'outputs/analysis/local_training_loss_summary_20261020'


def write_csv(name, rows, fields=None):
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / name).open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def read_json(p):
    return json.loads(p.read_text()) if p.exists() else {}


def scan_event(path):
    """Keep all epoch losses and exact means of step metrics between epochs."""
    inventory = {}
    pending = {}
    epochs, step_epochs = [], []
    last_step = None
    old_epoch = None
    old_stats = {}
    next_progress = 250_000_000
    records = 0
    with path.open('rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        offset, size = 0, len(mm)
        while offset < size:
            if offset + 12 > size:
                raise ValueError('Truncated TFRecord header: ' + str(path))
            length = struct.unpack_from('<Q', mm, offset)[0]
            end = offset + 12 + length
            if end + 4 > size:
                raise ValueError('Truncated TFRecord payload: ' + str(path))
            e = event_pb2.Event.FromString(mm[offset+12:end])
            offset = end + 4
            records += 1
            if offset >= next_progress:
                print(f'  scanned {offset/1e6:.0f} / {size/1e6:.0f} MB', flush=True)
                next_progress += 250_000_000
            for v in e.summary.value:
                if not v.HasField('simple_value'):
                    # Current project logs scalar floats, not tensor summaries.
                    if v.HasField('tensor'):
                        raise ValueError('Unexpected tensor scalar: ' + v.tag)
                    continue
                tag, value, step = v.tag, float(v.simple_value), int(e.step)
                s = inventory.get(tag)
                if s is None:
                    s = inventory[tag] = dict(samples=0, first_step=step, last_step=step,
                        first_value=value, last_value=value, minimum=value, maximum=value,
                        nonfinite=0, nonzero=0)
                s['samples'] += 1
                s['last_step'], s['last_value'] = step, value
                if math.isfinite(value):
                    s['minimum'], s['maximum'] = min(s['minimum'], value), max(s['maximum'], value)
                    s['nonzero'] += value != 0
                else:
                    s['nonfinite'] += 1
                if tag == 'debug/epoch':
                    old_epoch = int(value)
                if old_epoch is not None:
                    a = old_stats.setdefault((old_epoch,tag), [0,0.,0.,step,step])
                    a[0] += 1
                    delta = value-a[1]
                    a[1] += delta/a[0]
                    a[2] += delta*(value-a[1])
                    a[4] = step
                elif tag.startswith(('metric/', 'train_step/')):
                    last_step = step
                    # Welford statistics, including all recorded batches.
                    a = pending.setdefault(tag, [0, 0., 0., step, step])
                    a[0] += 1
                    delta = value - a[1]
                    a[1] += delta/a[0]
                    a[2] += delta*(value-a[1])
                    a[4] = step
                elif tag.startswith(('loss_', 'val_')):
                    epochs.append(dict(tag=tag, epoch0=step, value=value, wall_time=e.wall_time))
                    if tag == 'loss_total':
                        for t, a in pending.items():
                            step_epochs.append(dict(tag=t, epoch0=step, value=a[1], samples=a[0],
                                sample_SD=math.sqrt(max(0, a[2]/(a[0]-1))) if a[0]>1 else None,
                                first_step=a[3], last_step=a[4], origin='recorded_step_mean'))
                        pending = {}
    old_rows = [dict(epoch0=ep,tag=tag,value=a[1],samples=a[0],
                    sample_SD=math.sqrt(max(0,a[2]/(a[0]-1))) if a[0]>1 else None,
                    first_step=a[3],last_step=a[4],origin='logged_step_mean_by_debug_epoch')
                for (ep,tag),a in old_stats.items()]
    return dict(version=2, file=str(path.relative_to(ROOT)), bytes=path.stat().st_size,
        mtime_ns=path.stat().st_mtime_ns, records=records, inventory=inventory,
        epochs=epochs, step_epochs=step_epochs, old_step_epochs=old_rows,last_step=last_step,
        unfinished_step_metrics={k:dict(samples=a[0], mean=a[1], first_step=a[3],last_step=a[4])
                                 for k,a in pending.items()})


def extract():
    cache = OUT / 'event_cache'
    cache.mkdir(parents=True, exist_ok=True)
    copies = defaultdict(list)
    for d in sorted(BASE.iterdir()):
        if d.is_dir():
            for p in sorted(d.rglob('events.out.tfevents.*')):
                copies[p.name].append(p)
    scanned = {}
    for i, (name, paths) in enumerate(sorted(copies.items())):
        p = paths[0]
        # Duplicate event filenames must actually contain identical bytes.
        if len(paths)>1:
            digests = []
            for copy in paths:
                h = hashlib.sha256()
                with copy.open('rb') as f:
                    for b in iter(lambda:f.read(8*1024*1024), b''):
                        h.update(b)
                digests.append(h.hexdigest())
            if len(set(digests)) != 1:
                raise ValueError('Nonidentical copies: '+name)
        dest = cache / (name+'.json')
        existing = read_json(dest)
        if existing.get('version') == 2 and existing.get('bytes') == p.stat().st_size and existing.get('mtime_ns') == p.stat().st_mtime_ns:
            result = existing
        else:
            start = time.monotonic()
            print(f'[{i+1}/{len(copies)}] {name}: {p.stat().st_size/1e6:.1f} MB', flush=True)
            result = scan_event(p)
            dest.write_text(json.dumps(result, indent=2))
            print(f'  {len(result["epochs"])} epoch scalars; {len(result["inventory"])} tags; '
                  f'{time.monotonic()-start:.1f}s', flush=True)
        result['copies'] = [str(p.relative_to(ROOT)) for p in paths]
        scanned[name] = result
    (OUT / 'events_audit.json').write_text(json.dumps(scanned, indent=2))
    return scanned


DESCRIPTIONS = {
    'fmsr_flow_cr10_bs32_20260930_091149': ('FMSR CR fine-tune', 'FMSR', '在已训练31轮的FMSR权重上另做CR微调；不是从TF++开始的独立10轮训练。'),
    'fmsr_k60': ('Config only', 'config', '只有配置，没有TensorBoard训练记录；不能确认实际训练。'),
    'fmsr_model0002_joint_balanced_lr1e5_seed0_20261004_225835': ('Joint balanced recovery', 'recovery', '从3轮recovery模型开始；均衡recovery数据采样10%，CR关闭。'),
    'fmsr_pdm_lite_flow_fixed_bs64_20260929_092208': ('PDM flow fixed BS64', 'PDM', 'TF++初始化；目录标记flow_fixed版本，batch64；修复内容不能仅凭名称确定。'),
    'fmsr_pdm_lite_residual_flow_k60_20260926_142355': ('PDM flow BS16', 'PDM', 'TF++初始化；原始10轮PDM flow；原始events保存在续训目录中，已找回。'),
    'fmsr_pdm_lite_residual_flow_k60_resume10to14_20260927_095359': ('PDM resume E11-15 BS16', 'resume', '接原始epoch09续训；本阶段学习率3e-5，只有5轮，不把复制的前10轮当成本设置的数据。'),
    'fmsr_pdm_lite_residual_flow_k60_resume15to29_bs64_20260927_211110': ('PDM resume E16-30 BS64', 'resume', '接epoch14续训；本阶段改batch64，学习率3e-5。'),
    'fmsr_pdm_lite_tie_cr_20260930_234244': ('Early Jacobian CR', 'PDM', '早期endpoint_jacobian；CR权重1e-4、fraction0.0625、interval8。'),
    'fmsr_pdm_lite_tie_cr_epoch05_snapshot_20261001': ('Epoch05 snapshot', 'snapshot', '上述早期CR训练epoch05权重快照，不是独立训练。'),
    'fmsr_pdm_lite_tie_cr_fd_epoch11_recovery_seed0_20261003_181046': ('Recovery first 3 epochs', 'recovery', '从旧FD epoch11权重开始；recovery数据采样25%，CR关闭；采样恢复数据不等于flow_recovery_weight损失。'),
    'fmsr_pdm_lite_tie_cr_fd_recovery_resume3to22_20261004_230708': ('Recovery resume E4-23', 'resume', '接3轮recovery模型；recovery数据25%，CR关闭，余弦周期改为20。'),
    'fmsr_pdm_lite_tie_cr_fd_seed0': ('FD CR warm-up', 'PDM', '旧endpoint_fd；warmup1000，CR权重0.01、interval4。'),
    'fmsr_pdm_lite_tie_cr_fd_tau_delayed_seed0': ('FD CR delayed', 'PDM', '旧endpoint_fd；延迟18454步、渐增9227步、分段kappa；旧版首层训练问题见历史审计。'),
    'fmsr_pdm_lite_tie_cr_v2_seed0': ('Jacobian CR v2 + recovery', 'PDM', 'endpoint_jacobian权重0.001，额外flow_recovery_weight=0.05；与仅FM+CR目标不同。'),
    'flow_only_10epochs_seed0_20261008_140012': ('FM failed: head mismatch', 'failed', '启动失败：flow_only与恢复checkpoint的tie_cr_fd头不兼容；没有训练loss。'),
    'flow_only_10epochs_seed0_20261008_142537': ('FM failed: proxy', 'failed', '启动失败：预训练模型下载代理连接失败；没有训练loss。'),
    'flow_only_10epochs_seed0_20261008_143915': ('FM final', 'PDM', '当前闭环FM；flow_only，CR关闭，TF++初始化，10轮余弦退火。'),
    'posttrain_fmsr_k60_4090_32_residual_flow_20260917_154223_42965': ('FMSR early attempt', 'FMSR', '只有epoch0的一小段batch日志，不能当成完整epoch均值。'),
    'posttrain_fmsr_k60_4090_32_residual_flow_20260917_160256_45841': ('FMSR flow baseline', 'FMSR', 'FMSR训练管线与fmsr anchors；日志在backup中，配置曾被续训覆盖；历史初始权重不能仅凭现配置判定。'),
    'tie_spectral_10epochs_seed0_20261006_223036': ('FM+CR final spectral', 'PDM', '当前闭环FM+CR；endpoint_jacobian，权重0.01、fraction0.25、interval4；延迟2轮、渐增1轮，10轮余弦退火。'),
}


def build(scanned):
    import numpy as np
    runs, rows, inventories, provenance = [], [], [], []
    directories = sorted(d for d in BASE.iterdir() if d.is_dir())
    by_name = {d.name: f'R{i:02d}' for i,d in enumerate(directories, 1)}
    special = {
        '1790403947': 'fmsr_pdm_lite_residual_flow_k60_20260926_142355',
        '1790474115': 'fmsr_pdm_lite_residual_flow_k60_resume10to14_20260927_095359',
        '1790514759': 'fmsr_pdm_lite_residual_flow_k60_resume15to29_bs64_20260927_211110',
    }
    owned = defaultdict(list)
    for name, ev in scanned.items():
        timestamp = name.split('.')[3]
        owner = special.get(timestamp, Path(ev['file']).relative_to('outputs/local_training').parts[0])
        owned[owner].append((name, ev))
        provenance.append(dict(event_file=name, owner=by_name[owner], owner_directory=owner,
            bytes=ev['bytes'], copies=len(ev['copies']), paths=' | '.join(ev['copies']),
            nonfinite=sum(v['nonfinite'] for v in ev['inventory'].values())))
        for tag, values in ev['inventory'].items():
            inventories.append(dict(run=by_name[owner], event_file=name, tag=tag, **values))
    for d in directories:
        rid = by_name[d.name]
        config_paths = list(d.glob('config.json')) or sorted(d.glob('*/config.json'))
        config = read_json(config_paths[0]) if config_paths else {}
        args = read_json(d/'args.json') or read_json(d/'args.txt')
        variant = read_json(d/'variant.json')
        def setting(key, vkey=None, default=None):
            return config.get(key, args.get(key, variant.get(vkey or key.upper(), default)))
        label, family, note = DESCRIPTIONS[d.name]
        roots = setting('data_roots', default=[]) or []
        if isinstance(roots, str):
            roots = [roots]
        run = dict(run=rid, label=label, directory=d.name, family=family, note=note,
            config_file=str(config_paths[0].relative_to(ROOT)) if config_paths else '',
            batch_size=setting('batch_size'), lr=setting('lr','LEARNING_RATE'), seed=setting('seed'),
            requested_epochs=setting('epochs'), continue_epoch=setting('continue_epoch',default=0),
            freeze_backbone=setting('freeze_backbone'),
            head=setting('fmsr_trajectory_head',default='unspecified'),
            generator=setting('trajectory_generator'),
            init_checkpoint=setting('load_file','TFPP_CHECKPOINT'),
            CR_enabled=setting('flow_contract_enable'), CR_method=setting('flow_contract_method',default='unspecified'),
            CR_weight=setting('flow_contract_weight'), CR_fraction=setting('flow_contract_fraction'),
            CR_interval=setting('flow_contract_interval'), CR_warmup=setting('flow_contract_warmup_steps'),
            CR_delay=setting('flow_contract_delay_steps'), CR_ramp=setting('flow_contract_ramp_steps'),
            CR_tau_kappa=json.dumps(setting('flow_contract_tau_kappa')),
            flow_recovery_weight=setting('flow_recovery_weight'),
            recovery_data_fraction=setting('fmsr_recovery_fraction'),
            recovery_data_root=setting('fmsr_recovery_root'),
            frame_manifest=setting('fmsr_frame_manifest'),
            recovery_manifest=setting('fmsr_recovery_manifest'),
            recovery_sampling_plan=setting('fmsr_recovery_sampling_plan'),
            train_sampling_rate=setting('train_sampling_rate'),
            data_setting=setting('setting'),
            scheduler_T0=setting('cosine_t0'), scheduler_Tmult=setting('cosine_t_mult'),
            dataset_root_count=len(roots),
            dataset_names=' | '.join(sorted(Path(p).name for p in roots)),
            data_root_or_base=setting('carla_data') or variant.get('PDM_DATA_ROOT',''),
            anchor_path=setting('fmsr_anchor_path') or setting('plan_anchor_path'),
            event_files=' | '.join(n for n,_ in owned[d.name]))
        for name, ev in owned[d.name]:
            for r in ev['epochs']:
                rows.append(dict(run=rid, event_file=name, **r, samples=1,
                    sample_SD=None, origin='logged_epoch_mean', complete_epoch=True))
            for r in ev['step_epochs']:
                rows.append(dict(run=rid, event_file=name, **r, complete_epoch=True))
            # Old FMSR pipeline logs every batch with explicit debug/epoch.
            old = {(r['epoch0'],r['tag']):r for r in ev.get('old_step_epochs',[])}
            eps = sorted({k[0] for k in old})
            for ep in eps:
                epoch_tag = old[ep,'debug/epoch']
                expected = int(round(old[ep,'debug/dataloader_size']['value']))
                complete = epoch_tag['samples'] == expected
                for (e,tag),r in old.items():
                    if e == ep:
                        rows.append(dict(run=rid,event_file=name,**r,complete_epoch=complete))
                scaled = [r for (e,t),r in old.items() if e == ep and t.startswith('scaled_loss/')]
                if scaled:
                    assert len({r['samples'] for r in scaled}) == 1
                    rows.append(dict(run=rid,event_file=name,tag='loss_total',epoch0=ep,
                        value=sum(r['value'] for r in scaled),samples=scaled[0]['samples'],
                        sample_SD=None,origin='sum_of_scaled_component_step_means',complete_epoch=complete))
                    for new,oldtag in [('loss_fmsr_reg','scaled_loss/loss_ddbm_reg'),
                                       ('loss_fmsr_cls','scaled_loss/loss_ddbm_cls')]:
                        if (ep,oldtag) in old:
                            r=old[ep,oldtag]
                            rows.append(dict(r,run=rid,event_file=name,tag=new,
                                complete_epoch=complete,origin='scaled_FMSR_loss_ddbm_alias'))
        # No copied/inherited epochs are silently assigned to new settings.
        own_totals = sorted((r for r in rows if r['run']==rid and r['tag']=='loss_total'
                             and r['complete_epoch']),key=lambda r:r['epoch0'])
        indices=[r['epoch0'] for r in own_totals]
        assert len(indices)==len(set(indices)), ('Overlapping epoch totals',rid,indices)
        run['completed_epochs']=len(indices)
        run['first_epoch0']=min(indices) if indices else None
        run['last_epoch0']=max(indices) if indices else None
        run['has_global_first10']=all(i in indices for i in range(10))
        run['has_stage_first10']=len(indices)>=10 and indices[:10]==list(range(indices[0],indices[0]+10))
        run['status']=('no_training_log' if not owned[d.name] else 'batch_only_incomplete' if not indices
                       else '10_epochs_available' if run['has_stage_first10'] else 'fewer_than_10_epochs')
        batch_counts=sorted({r['samples'] for r in rows if r['run']==rid and r['complete_epoch']
                            and r['tag'] in ('train_step/loss_fmsr_reg','scaled_loss/loss_ddbm_reg')})
        run['recorded_batches_per_epoch']=' | '.join(map(str,batch_counts))
        sizes=[r['value'] for r in rows if r['run']==rid and r['tag']=='debug/dataset_size']
        run['logged_dataset_size']=int(round(sizes[0])) if sizes else None
        if family=='snapshot':run['status']='snapshot_not_independent'
        if family=='failed':run['status']='failed_before_training'
        run['snapshot_source']=read_json(d/'snapshot_source.json').get('source_dir','')
        runs.append(run)
    # For all stage curves, epoch1 is the first epoch of this setting, not always global epoch0.
    run_lookup={r['run']:r for r in runs}
    for r in rows:
        start=run_lookup[r['run']]['first_epoch0']
        r['epoch']=r['epoch0']+1
        r['stage_epoch']=r['epoch0']-(start if start is not None else r['epoch0'])+1
    write_csv('run_settings.csv',runs)
    write_csv('event_provenance.csv',provenance)
    write_csv('scalar_inventory.csv',inventories)
    write_csv('all_epoch_metrics_long.csv',rows)
    loss_rows=[r for r in rows if 'loss' in r['tag'].lower()]
    write_csv('all_losses_long.csv',loss_rows)
    write_csv('first10_epoch_losses_long.csv',[r for r in loss_rows if r['epoch0']<10])
    write_csv('first10_stage_losses_long.csv',[r for r in loss_rows if r['stage_epoch']<=10])
    # A tidy, compact table of the substantive objectives and fitting metrics.
    canonical={
        'total_loss':['loss_total'], 'reg_loss':['loss_fmsr_reg'],
        'cls_loss':['loss_fmsr_cls'],
        'flow_loss':['metric/flow_fm_loss','metric/flow_velocity_l1'],
        'CR_loss':['metric/flow_contract_loss'],
        'recovery_loss':['metric/flow_recovery_loss'],
        'endpoint_L1':['metric/flow_endpoint_l1'],
        'train_ADE':['metric/route_ade'],'train_FDE':['metric/route_fde'],
        'val_total_loss':['val_loss_total'],
    }
    lookup={(r['run'],r['epoch0'],r['tag']):r for r in rows if r['complete_epoch']}
    compact=[]
    for run in runs:
        for ep in sorted({r['epoch0'] for r in rows if r['run']==run['run'] and r['tag']=='loss_total' and r['complete_epoch']}):
            r=dict(run=run['run'],label=run['label'],epoch0=ep,epoch=ep+1,
                   stage_epoch=ep-run['first_epoch0']+1)
            for short,tags in canonical.items():
                src=next((lookup[run['run'],ep,t] for t in tags if (run['run'],ep,t) in lookup),None)
                r[short]=src['value'] if src else None
            compact.append(r)
    write_csv('epoch_loss_summary.csv',compact)
    for axis,name in [('epoch','total_loss_first10_global_wide.csv'),('stage_epoch','total_loss_first10_stage_wide.csv')]:
        wide=[]
        for run in runs:
            r=dict(run=run['run'],label=run['label'],status=run['status'],first_epoch0=run['first_epoch0'])
            for i in range(1,11):
                vals=[v['total_loss'] for v in compact if v['run']==run['run'] and v[axis]==i]
                r[f'epoch_{i:02d}']=vals[0] if vals else None
            wide.append(r)
        write_csv(name,wide)
    summary=[]
    for run in runs:
        all_samples=sorted((r for r in compact if r['run']==run['run']),key=lambda r:r['stage_epoch'])
        samples=[r for r in all_samples if r['stage_epoch']<=10]
        s=dict(run=run['run'],label=run['label'],epochs_available=run['completed_epochs'],
               comparison_first_global_epoch=run['first_epoch0'],complete_10=run['has_stage_first10'],
               first_total=samples[0]['total_loss'] if samples else None,
               last_available_total=all_samples[-1]['total_loss'] if all_samples else None,
               last_available_epoch0=all_samples[-1]['epoch0'] if all_samples else None,
               epoch10_total=samples[-1]['total_loss'] if len(samples)==10 else None,
               mean_first10_total=float(np.mean([r['total_loss'] for r in samples])) if len(samples)==10 else None)
        for metric in canonical:
            s['epoch10_'+metric]=samples[-1][metric] if len(samples)==10 else None
        summary.append(s)
    write_csv('first10_summary.csv',summary)
    lineages=[]
    for name,ids in [('PDM flow original and resumes',['R05','R06','R07']),
                     ('Recovery first stage and resume',['R10','R11'])]:
        for r in compact:
            if r['run'] in ids:
                lineages.append(dict(lineage=name,**r))
    write_csv('lineage_epoch_loss_summary.csv',lineages)
    write_csv('lineage_first10_loss_summary.csv',[r for r in lineages if r['epoch']<=10])
    partial=[]
    for name,ev in scanned.items():
        owner=next(r['owner'] for r in provenance if r['event_file']==name)
        for tag,values in ev['unfinished_step_metrics'].items():
            partial.append(dict(run=owner,event_file=name,tag=tag,origin='unfinished_epoch_batches',**values))
        for r in rows:
            if r['event_file']==name and not r['complete_epoch']:
                partial.append(dict(run=owner,event_file=name,tag=r['tag'],epoch0=r['epoch0'],
                    origin=r['origin'],samples=r['samples'],mean=r['value'],
                    first_step=r.get('first_step'),last_step=r.get('last_step')))
    write_csv('unfinished_epoch_metrics.csv',partial)
    (OUT/'summary.json').write_text(json.dumps(dict(runs=runs,summary=summary,epoch_metrics=compact),indent=2))
    validate(scanned,rows,compact)
    plot_all(runs,compact,rows,scanned)
    report(runs,summary,compact,scanned)
    print('Wrote analysis:',OUT,flush=True)


def validate(scanned,rows,compact):
    """Cross-check independent existing exports and the standard TFRecord reader."""
    from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
    name=min(scanned,key=lambda n:scanned[n]['bytes'])
    standard=defaultdict(int)
    for raw in RawEventFileLoader(str(ROOT/scanned[name]['file'])).Load():
        ev=event_pb2.Event.FromString(raw)
        for v in ev.summary.value:
            if v.HasField('simple_value'):standard[v.tag]+=1
    assert dict(standard)=={k:v['samples'] for k,v in scanned[name]['inventory'].items()}
    previous=ROOT/'outputs/analysis/fm_cr_training_curves_20261019/epoch_metrics.csv'
    checks=0
    if previous.exists():
        prior=list(csv.DictReader(previous.open()))
        # The old export's exact field names are checked below when available.
        for model,label in [('FM','FM final'),('FM+CR','FM+CR final spectral')]:
            for r in compact:
                if r['label']!=label:continue
                for p in prior:
                    if (p.get('model')==model and p.get('bin_type')=='epoch'
                            and int(p.get('bin_index',-1))==r['epoch0']
                            and p.get('metric')=='metric/flow_fm_loss'):
                        assert math.isclose(r['flow_loss'],float(p['mean']),rel_tol=1e-7,abs_tol=1e-9)
                        checks+=1
    for label,expect in [('FM final',.025893967599),('FM+CR final spectral',.03376267478)]:
        record=next(r for r in compact if r['label']==label and r['epoch']==10)
        assert math.isclose(record['total_loss'],expect,rel_tol=1e-6)
    assert not any(not math.isfinite(r['value']) for r in rows)
    (OUT/'validation.json').write_text(json.dumps(dict(standard_reader_file=name,
        standard_reader_tag_counts_match=True,prior_epoch_crosschecks=checks,
        known_final_totals_match=True,all_aggregated_values_finite=True),indent=2))


def plot_all(runs,compact,rows,scanned):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False})
    valid=[r for r in runs if r['completed_epochs']]
    colors={'total_loss':'#333333','reg_loss':'#4C8997','cls_loss':'#C76870',
            'flow_loss':'#3679AD','CR_loss':'#B58A44','recovery_loss':'#976DA1'}
    # Readable numerical table includes all directories, including missing data.
    fig,ax=plt.subplots(figsize=(16,9))
    ax.axis('off')
    cells=[]
    for run in runs:
        row=[run['run'],run['label']]
        for ep in range(1,11):
            v=next((r['total_loss'] for r in compact if r['run']==run['run'] and r['epoch']==ep),None)
            row.append('—' if v is None else f'{v:.4f}')
        cells.append(row)
    table=ax.table(cellText=cells,colLabels=['ID','Training setting']+[f'E{i:02d}' for i in range(1,11)],
        colWidths=[.04,.26]+[.07]*10,cellLoc='center',loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1,1.7)
    for (row,col),cell in table.get_celld().items():
        cell.set_edgecolor('#D9DEE2')
        if row==0:cell.set_facecolor('#E1EBF0');cell.set_text_props(weight='bold')
        elif row%2==0:cell.set_facecolor('#F4F7F9')
        if col==1:cell.set_text_props(ha='left')
    ax.set_title('Total training loss: original epochs 1–10\nMissing and pre-resume epochs are left blank; snapshots are not independent runs. Different objectives are not a ranking.',fontsize=13,pad=10)
    fig.tight_layout()
    fig.savefig(OUT/'total_loss_first10_table.png',dpi=400,bbox_inches='tight')
    fig.savefig(OUT/'total_loss_first10_table.pdf',bbox_inches='tight')
    plt.close(fig)
    def decorate(ax):
        ax.grid(alpha=.2)
        ax.set_xlim(.7,10.3)
        ax.set_xticks([1,2,4,6,8,10])
        ax.set_xlabel('Epoch within this training stage')
        ax.set_ylabel('Loss (log scale)')
        ax.set_yscale('log')
    # Small multiples expose interrupted and resumed settings without overcrowding.
    fig,axes=plt.subplots(math.ceil(len(valid)/3),3,figsize=(15,3.2*math.ceil(len(valid)/3)),squeeze=False)
    for ax,run in zip(axes.flat,valid):
        values=[r for r in compact if r['run']==run['run'] and r['stage_epoch']<=10]
        for key in colors:
            v=[r for r in values if r[key] is not None and r[key]>0]
            if v:ax.plot([r['stage_epoch'] for r in v],[r[key] for r in v],'.-',color=colors[key],label=key.replace('_',' '),ms=4,lw=1.3)
        ax.set_title(run['run']+'  '+run['label'],fontsize=10)
        decorate(ax)
        ax.legend(fontsize=7,ncol=2,loc='best')
        if run['first_epoch0']:
            ax.text(.02,.91,f"Global epochs {run['first_epoch0']+1}+",transform=ax.transAxes,fontsize=8)
    for ax in list(axes.flat)[len(valid):]:ax.axis('off')
    fig.suptitle('All local training stages: first 10 available epochs\nDifferent datasets, objectives and initializations; loss magnitudes are not a performance ranking.',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.96))
    fig.savefig(OUT/'all_training_losses_first10.png',dpi=400,bbox_inches='tight')
    fig.savefig(OUT/'all_training_losses_first10.pdf',bbox_inches='tight')
    plt.close(fig)
    # A compact final-model comparison with the shared FM term separated from total.
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for ax,key in zip(axes.flat,['total_loss','flow_loss','cls_loss','CR_loss','train_ADE','train_FDE']):
        for label,color in [('FM final','#4C8997'),('FM+CR final spectral','#C76870')]:
            v=sorted((r for r in compact if r['label']==label and r['epoch']<=10 and r[key] is not None),key=lambda r:r['epoch'])
            if v:ax.plot([r['epoch'] for r in v],[r[key] for r in v],'o-',color=color,ms=4,label=label)
        ax.set_title(key.replace('_',' '))
        ax.set_xlabel('Epoch')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Evaluated FM and FM+CR: training diagnostics (not validation / closed-loop performance)')
    fig.tight_layout(rect=(0,0,1,.95))
    fig.savefig(OUT/'final_FM_vs_CR_first10.png',dpi=400,bbox_inches='tight')
    fig.savefig(OUT/'final_FM_vs_CR_first10.pdf',bbox_inches='tight')
    plt.close(fig)
    # All nonzero loss tags, per run, including per-epoch validation if present.
    with PdfPages(OUT/'all_loss_curves_by_setting.pdf') as pdf:
        for run in valid:
            subset=[r for r in rows if r['run']==run['run'] and r['complete_epoch'] and r['stage_epoch']<=10 and 'loss' in r['tag'].lower()]
            tags=sorted({r['tag'] for r in subset if r['value']!=0})
            for start in range(0,len(tags),9):
                fig,axes=plt.subplots(3,3,figsize=(14,10))
                for ax,tag in zip(axes.flat,tags[start:start+9]):
                    v=sorted((r for r in subset if r['tag']==tag),key=lambda r:r['stage_epoch'])
                    ax.plot([r['stage_epoch'] for r in v],[r['value'] for r in v],'.-',color='#4C8997')
                    ax.set_title(tag,fontsize=8)
                    ax.grid(alpha=.2)
                    ax.set_xlabel('Stage epoch')
                for ax in list(axes.flat)[len(tags[start:start+9]):]:ax.axis('off')
                fig.suptitle(run['run']+' '+run['label']+' | all nonzero recorded loss tags')
                fig.tight_layout(rect=(0,0,1,.96))
                pdf.savefig(fig)
                plt.close(fig)
    # Preserve curves of runs interrupted before completing even one epoch.
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    partial_runs=[r for r in runs if r['status']=='batch_only_incomplete']
    if partial_runs:
        fig,axes=plt.subplots(len(partial_runs),2,figsize=(12,3.3*len(partial_runs)),squeeze=False)
        for row,run in zip(axes,partial_runs):
            name=run['event_files'].split(' | ')[0]
            events=EventAccumulator(str(ROOT/scanned[name]['file']),size_guidance={'scalars':0})
            events.Reload()
            for ax,choices,title in zip(row,[['train_step/loss_fmsr_reg','unscaled_loss/loss_ddbm_reg'],
                ['train_step/loss_fmsr_cls','unscaled_loss/loss_ddbm_cls']],['Regression loss','Classification loss']):
                tag=next((t for t in choices if t in events.Tags()['scalars']),None)
                if tag:
                    v=events.Scalars(tag)
                    steps=np.array([r.step for r in v]);values=np.array([r.value for r in v])
                    bins=steps//100
                    ids=np.unique(bins)
                    ax.plot([steps[bins==i].mean() for i in ids],[values[bins==i].mean() for i in ids],'.-',color='#4C8997')
                ax.set_title(run['run']+' '+run['label']+' | '+title,fontsize=10)
                ax.set_xlabel('Logged step (100-step block means)')
                ax.grid(alpha=.2)
        fig.suptitle('Incomplete attempts: less than one epoch; excluded from epoch-mean comparisons')
        fig.tight_layout(rect=(0,0,1,.95))
        fig.savefig(OUT/'incomplete_attempts_batch_losses.png',dpi=400,bbox_inches='tight')
        fig.savefig(OUT/'incomplete_attempts_batch_losses.pdf',bbox_inches='tight')
        plt.close(fig)


def report(runs,summary,compact,scanned):
    def f(v):return '—' if v is None else f'{v:.6f}'
    text=['# local_training 全设置训练loss审计', '',
        '汇总日期：2026-10-20。只读原始配置和TensorBoard事件；未修改数据、权重，未启动训练。', '',
        '## 范围和对齐', '',
        f'- 发现 {len(runs)} 个目录、{sum(len(v["copies"]) for v in scanned.values())} 份事件文件副本、{len(scanned)} 份唯一事件文件。重复文件已逐字节SHA256核对并去重。',
        f'- {sum(r["completed_epochs"]>0 for r in runs)} 个训练阶段有完整epoch记录；{sum(r["has_global_first10"] for r in runs)} 个设置有原始epoch00–09；{sum(r["has_stage_first10"] for r in runs)} 个阶段有连续10轮（含从较晚epoch开始的续训阶段）。',
        '- epoch列从1开始显示，对应checkpoint epoch0从0开始。续训目录只计本阶段新增的epoch；stage_epoch=1表示本阶段第一轮，可能是原始第11/16轮。',
        '- 主汇总不补全、不外推未完成的epoch；缺失为留空。快照不算独立训练，失败启动不产生loss曲线。',
        '- PDM loss_*是训练器记录的整轮batch均值。metric/*按每个loss_total之前的实际记录batch分组，不假设所有设置都是9227步/轮。',
        '- FMSR旧管线没有epoch loss_total：按debug/epoch聚合全部已记录batch，total=各scaled_loss分量均值之和。只有记录batch数等于debug/dataloader_size时才计完整epoch；不完整的batch均值仍在长表中并标complete_epoch=False。',
        '- 图中仅展示非零loss；所有为零的loss标签仍保存在all_losses_long.csv和scalar_inventory.csv。未给不同epoch的均值套统计误差线。', '',
        '## 设置与完成度', '',
        '| ID | 设置 | batch | 每轮记录batch数 | 初始LR | CR方法/权重 | 记录到的完整epoch（原始编号从0起） | 特性 |',
        '|---|---|---:|---:|---:|---|---|---|']
    for r in runs:
        epochs='无' if not r['completed_epochs'] else f'{r["first_epoch0"]}–{r["last_epoch0"]}（{r["completed_epochs"]}轮）'
        text.append(f'| {r["run"]} | {r["label"]} | {r["batch_size"]} | {r["recorded_batches_per_epoch"] or "—"} | {r["lr"]} | {r["CR_method"]} / {r["CR_weight"]}，enable={r["CR_enabled"]} | {epochs} | {r["note"]} |')
    text+=['','## 前10轮loss汇总','','此表仅有连续10轮时填入第10轮及10轮均值；续训是本阶段10轮，不能冒充从头训练的10轮。', '',
           '| ID | 设置 | 总loss第1轮 | 总loss第10轮 | 前10轮总loss均值 | FM项第10轮 | CR项第10轮 |',
           '|---|---|---:|---:|---:|---:|---:|']
    for s in summary:
        text.append(f'| {s["run"]} | {s["label"]} | {f(s["first_total"])} | {f(s["epoch10_total"])} | {f(s["mean_first10_total"])} | {f(s["epoch10_flow_loss"])} | {f(s["epoch10_CR_loss"])} |')
    text+=['','## 曲线特性与初步结果','',
        '- 当前FM（R17）：第1→10轮总loss 0.115249→0.025894，下降77.53%；FM项第10轮0.024438。',
        '- 当前谱CR（R20）：第1→10轮总loss 0.113751→0.033763，下降70.32%；FM项第10轮0.030776。两者总loss和共享FM项都下降，但不能由此主张CR的拟合loss更低。',
        '- R20的CR项在延迟后开始记录，第3轮0.468909，第4轮0.125727，第10轮0.152920；后段并未持续下降，总loss下降不代表CR约束已充分满足。',
        '- 原始PDM flow R05（BS16）第10轮总loss0.148388；flow_fixed R04（BS64）为0.171406。batch及实现版本不同，不能把差异仅归因于batch大小。',
        '- 早期Jacobian CR R08只有7个完整epoch；总loss0.277233→0.225935，同时记录CR项3.435898→6.186021。小权重下总拟合改善可以与未加权CR项增大同时发生。',
        '- R01仅3轮、R03仅3轮、R06仅5轮、R10仅3轮、R14仅1轮。R12与R18只有未完成首轮的batch曲线；这些都不足以填入第10轮结果。',
        '- R05→R06→R07及R10→R11的连续训练轨迹另存lineage表。后者可恢复recovery阶段原始前10轮，但其中第4轮起余弦周期配置已变化，不是一个固定设置的独立10轮实验。']
    text+=['- 实际每轮batch数不同：正式R17/R20均9227；R05为18454（BS16）、R04为4613（BS64）；recovery R10/R11为1696、joint balanced R03为403。后两类训练的有效数据/采样规模与正式训练不同，不能把它们的一轮当成相同训练量。FMSR旧管线完整轮均30272 batch。']
    text+=['','## 如何解读','','1. 同一设置内，下降曲线可以说明训练拟合/优化的变化；绝对loss不能对所有设置直接排名。FMSR和PDM的数据、anchor及实现不同；恢复采样、初始化权重、batch、学习率和余弦周期也有变化。',
        '2. FM与FM+CR应拆开看共享FM项、分类项和CR项。reg/total含额外正则时，其值高于纯FM并不等于闭环驾驶更差；训练loss也不能证明泛化或鲁棒性优势。',
        '3. 旧endpoint_fd与新endpoint_jacobian的CR定义不同，不能把两种CR loss大小当成同一尺度的约束强弱。目录/head名tie_cr_fd不保证方法是FD，应看flow_contract_method。',
        '4. 历史审计发现旧FD首层没有优化器更新、新谱版本已修复，同时余弦周期从20改为10。新旧拟合改善不能单独归因于CR。详见outputs/training_comparison/20261007/tie_spectral_vs_fd_epoch09/report.md。',
        '5. recovery数据采样比例与flow_recovery_weight是两个不同设置，已分别列出。恢复训练CR关闭也可能继承此前CR权重，不能当作从同一初始权重训练的FM基线。',
        '6. 训练ADE/FDE取训练batch预测端点/桥状态，不是独立验证或闭环DS/SR。',
        f'7. 本次事件中含val_标签的文件数：{sum(any(t.startswith("val_") for t in ev["inventory"]) for ev in scanned.values())}。验证标签若有则在长表及逐设置PDF保留。', '',
        '## 文件', '',
        '- all_training_losses_first10.png / .pdf：各设置前10轮总loss、回归、分类、FM/CR项小多图。',
        '- total_loss_first10_table.png / .pdf：20个目录原始前10轮总loss数值总表。',
        '- final_FM_vs_CR_first10.png / .pdf：当前闭环两模型的共享训练指标对比。',
        '- all_loss_curves_by_setting.pdf：所有设置、所有非零loss标签逐页绘图。',
        '- incomplete_attempts_batch_losses.png / .pdf：未完成首轮的两次尝试的batch loss曲线。',
        '- epoch_loss_summary.csv：所有完整epoch的主要loss和训练ADE/FDE。',
        '- total_loss_first10_global_wide.csv：所有目录在原始epoch00–09的总loss横表；续训阶段的旧epoch留空。',
        '- total_loss_first10_stage_wide.csv：所有阶段各自前10轮总loss横表；保留续训原始起点。',
        '- first10_summary.csv：前10轮统计；不足10轮时第10轮/10轮均值留空。',
        '- lineage_epoch_loss_summary.csv / lineage_first10_loss_summary.csv：两条原始训练及续训连续轨迹，保留每轮实际来源设置。',
        '- unfinished_epoch_metrics.csv：中断轮次已记录batch的汇总，不计完整epoch。',
        '- run_settings.csv：20个目录完整设置、数据根目录、anchor、初始化checkpoint和训练特性。',
        '- all_losses_long.csv / first10_epoch_losses_long.csv / first10_stage_losses_long.csv：全部loss（含零值、不完整轮、原始标签及来源）。',
        '- event_provenance.csv / scalar_inventory.csv / events_audit.json / validation.json：来源去重、记录量、非有限值检查、交叉核对。',
        '- scripts/pdm_lite/audit_local_training.py：可复现审计和绘图脚本。', '',
        '技术说明：快速读取器检查TFRecord边界并解析protobuf，未逐条校验CRC；另用TensorBoard标准读取器核对最小文件，并核对已有FM/CR逐轮导出和已知第10轮总loss。']
    (OUT/'REPORT.md').write_text('\n'.join(text)+'\n')
    links=['REPORT.md','total_loss_first10_table.png','all_training_losses_first10.png','final_FM_vs_CR_first10.png','all_loss_curves_by_setting.pdf',
           'run_settings.csv','epoch_loss_summary.csv','total_loss_first10_global_wide.csv','total_loss_first10_stage_wide.csv','first10_summary.csv']
    page='<meta charset="utf-8"><title>Local training loss audit</title><h1>训练loss全设置汇总</h1>'
    page+=''.join(f'<p><a href="{p}">{p}</a></p>' for p in links)
    page+='<img style="max-width:100%" src="all_training_losses_first10.png"><img style="max-width:100%" src="final_FM_vs_CR_first10.png">'
    page+='<pre style="white-space:pre-wrap">'+html.escape('\n'.join(text))+'</pre>'
    (OUT/'index.html').write_text(page)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    build(extract())
