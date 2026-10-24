"""Prepare RGB, telemetry, actor geometry and temporal evidence for label review."""
import argparse
from collections import defaultdict
import json
import math
import html
import sys
import gzip
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def ego_points(points, ego):
    xy = np.asarray(points, dtype=float)[:, :2] - np.asarray(ego['position_m'][:2])
    yaw = math.radians(ego['rotation_deg'][2])
    return xy @ np.array([[math.cos(yaw), -math.sin(yaw)],
                          [math.sin(yaw), math.cos(yaw)]])


def actor_geometry(row):
    ego = row['ego']
    result = []
    for actor in row['nearby_actors']:
        if not actor['type'].startswith(('vehicle.', 'walker.')) or actor['id'] == ego['id']:
            continue
        offset = actor.get('bbox_offset_m', [0, 0, 0])
        yaw = math.radians(actor['rotation_deg'][2])
        position = np.asarray(actor['position_m']) + [
            offset[0] * math.cos(yaw) - offset[1] * math.sin(yaw),
            offset[0] * math.sin(yaw) + offset[1] * math.cos(yaw), offset[2]]
        xy = ego_points([position], ego)[0]
        extent = actor.get('bbox_extent_m', [.5, .5, 1])
        relative = yaw - math.radians(ego['rotation_deg'][2])
        ex = abs(math.cos(relative)) * extent[0] + abs(math.sin(relative)) * extent[1]
        ey = abs(math.sin(relative)) * extent[0] + abs(math.cos(relative)) * extent[1]
        if abs(position[2] - ego['position_m'][2]) > 4:
            continue
        result.append({'id': actor['id'], 'type': actor['type'], 'x_m': float(xy[0]),
            'y_m': float(xy[1]), 'half_x_m': ex, 'half_y_m': ey,
            'yaw_rad': relative, 'extent': extent[:2],
            'speed_mps': float(np.linalg.norm(actor.get('velocity_mps', [0, 0, 0])[:2])),
            'front_gap_m': float(xy[0] - ex - ego['bbox_extent_m'][0]),
            'overlaps_ego_straight_corridor': bool(abs(xy[1]) <= ey + ego['bbox_extent_m'][1])})
    return result


def decision(speed):
    return 'stop' if speed < .5 else 'go' if speed > 2 else 'slow'


def distance_to_route(points, route):
    """Distance to the finite navigation polyline, including its endpoints."""
    points, route = np.asarray(points)[:, :2], np.asarray(route)[:, :2]
    if len(route) == 1:
        return np.linalg.norm(points - route[0], axis=1)
    segment = route[1:] - route[:-1]
    offset = points[:, None, :] - route[None, :-1, :]
    fraction = (offset * segment[None, :, :]).sum(axis=-1) / np.maximum((segment ** 2).sum(axis=-1), 1e-12)
    closest = route[None, :-1, :] + np.clip(fraction, 0, 1)[..., None] * segment[None, :, :]
    return np.linalg.norm(points[:, None, :] - closest, axis=-1).min(axis=1)


def persistence(frames):
    spans = {}
    start = 0
    for index in range(1, len(frames) + 1):
        previous = frames[index - 1]
        previous_kind = decision(previous['policy'].get('teacher_target_speed_mps', 0))
        boundary = index == len(frames)
        if not boundary:
            current = frames[index]
            boundary = (decision(current['policy'].get('teacher_target_speed_mps', 0)) != previous_kind
                        or current['simulation_frame'] != previous['simulation_frame'] + 1
                        or bool(current['policy'].get('label_valid')) != bool(previous['policy'].get('label_valid')))
        if boundary:
            duration = previous['agent_timestamp_s'] - frames[start]['agent_timestamp_s'] + .05
            for frame in frames[start:index]:
                spans[frame['simulation_frame']] = duration
            start = index
    return spans


def prepare(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = json.loads((source / 'event_samples_report.json').read_text())
    candidates = rows(source / 'candidate_samples.jsonl')
    route_inputs = {}
    for evaluation in report['evaluations']:
        evaluation = Path(evaluation)
        root = Path(json.loads((evaluation / 'collection_config.json').read_text())['collection_root'])
        for record in json.loads((evaluation / 'checkpoint_endpoint.json').read_text())['_checkpoint']['records']:
            diagnostic = root / 'diagnostics' / record['save_name']
            frames = rows(diagnostic / 'frames.jsonl')
            route_inputs[record['save_name'].replace('_rep0_', '_Rep0_')] = (
                frames, {r['simulation_frame']: r for r in frames}, persistence(frames),
                json.loads((diagnostic / 'metadata.json').read_text()), rows(diagnostic / 'events.jsonl'))
    grouped = defaultdict(list)
    for candidate in candidates:
        route = Path(candidate['measurement_directory']).parent
        frames, lookup, spans, metadata, events = route_inputs[route.name]
        frame = lookup[candidate['simulation_frame']]
        actors = actor_geometry(frame)
        leads = [actor for actor in actors if actor['x_m'] > 0 and actor['overlaps_ego_straight_corridor']]
        fmsr = min(leads, key=lambda actor: actor['front_gap_m']) if leads else None
        policy = frame['policy']
        teacher = np.asarray(policy['teacher_route_ego_m'])
        student = np.asarray(policy['predicted_trajectory_ego_m'])
        navigation = ego_points(metadata['route_world_m'], frame['ego'])
        nav_distances = distance_to_route(teacher[:6], navigation)
        nearby = [r for r in frames if abs(r['agent_timestamp_s'] - candidate['time_s']) <= .5]
        item = dict(candidate, route_id=route.name.split('_Rep0_')[0],
            teacher_decision=decision(candidate['teacher_target_speed_mps']),
            teacher_decision_persistence_s=spans[candidate['simulation_frame']],
            teacher_stop_fraction_pm_05s=float(np.mean([r['policy'].get('teacher_target_speed_mps', 0) < .5 for r in nearby])),
            fmsr_actor=fmsr, nearby_dynamic_actors=actors,
            nearby_traffic_lights=[a for a in frame['nearby_actors'] if a['type'].startswith('traffic.traffic_light')],
            trajectory_first6_distance_m=float(np.linalg.norm(teacher[:6] - student[:6], axis=1).mean()),
            teacher_trajectory_ego_m=teacher.tolist(), student_trajectory_ego_m=student.tolist(),
            navigation_route_ego_m=navigation.tolist(),
            teacher_first6_navigation_distance_m=float(nav_distances.mean()),
            ego=frame['ego'], route_completion_percent=frame['route_completion_percent'],
            rgb_path=str(route / 'rgb' / f"{candidate['frame']:04d}.jpg"))
        grouped[item['route_id']].append(item)
    evidence = [item for items in grouped.values() for item in items]
    with (output / 'frame_evidence.jsonl').open('w') as handle:
        for item in evidence:
            handle.write(json.dumps(item, allow_nan=False) + '\n')
    for route_id, items in grouped.items():
        for page_start in range(0, len(items), 24):
            page = items[page_start:page_start + 24]
            canvas = np.full((math.ceil(len(page) / 4) * 212, 4 * 384, 3), 245, np.uint8)
            for index, item in enumerate(page):
                x, y = index % 4 * 384, index // 4 * 212
                image = cv2.imread(item['rgb_path'])
                if image is None:
                    raise ValueError(item['rgb_path'])
                canvas[y:y + 160, x:x + 384] = cv2.resize(image, (384, 160))
                lines = [f"f{item['frame']} t{item['time_s']:.2f} V{item['measured_speed_mps']:.1f} S{item['student_target_speed_mps']:.1f} T{item['teacher_target_speed_mps']:.1f}",
                         f"{item['teacher_decision']} stable {item['teacher_decision_persistence_s']:.2f}s gap " +
                         (f"{item['fmsr_actor']['front_gap_m']:.1f}m" if item['fmsr_actor'] else '--'),
                         ','.join(item['reasons'])[:48]]
                for line_index, text in enumerate(lines):
                    cv2.putText(canvas, text, (x + 4, y + 176 + line_index * 15),
                                cv2.FONT_HERSHEY_SIMPLEX, .38, (20, 20, 20), 1, cv2.LINE_AA)
            path = output / f'{route_id}_page{page_start // 24 + 1}.jpg'
            cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(route_id, 'samples', len(items), 'stop', sum(i['teacher_decision'] == 'stop' for i in items),
              'go', sum(i['teacher_decision'] == 'go' for i in items),
              'stable_go', sum(i['teacher_decision'] == 'go' and i['teacher_decision_persistence_s'] >= .5 for i in items), flush=True)
    (output / 'review_source.json').write_text(json.dumps(report, indent=2) + '\n')
    return evidence


# Manually reviewed RGB windows from these five specific rollouts. These are
# review decisions, not a classifier to apply to other runs with the same IDs.
REVIEW_WINDOWS = {
    '2204': {'better': [(14, 17)], 'wait': [(18, 21), (745, 748)]},
    '2215': {'better': [(14, 16), (710, 712), (723, 724), (726, 748)],
             'wait': [(18, 20), (702, 704), (713, 748)]},
    '24816': {'better': [(10, 11), (18, 23), (44, 48)], 'wait': []},
    '2115': {'better': [(11, 17), (21, 21)], 'wait': [(19, 20), (25, 27)]},
    '2127': {'better': [(13, 15)], 'wait': [(17, 19)]},
}
REVIEWED_ROLLOUTS = {
    '2204': '10_04_20_38_34', '2215': '10_04_21_55_52',
    '24816': '10_04_22_15_28', '2115': '10_04_22_16_37',
    '2127': '10_04_22_18_11',
}


def in_windows(frame, windows):
    return any(start <= frame <= end for start, end in windows)


def finalize(evidence, output):
    output = Path(output)
    reviewed = []
    for item in evidence:
        route = item['route_id'].split('_')[-1]
        if not Path(item['measurement_directory']).parent.name.endswith(REVIEWED_ROLLOUTS[route]):
            raise ValueError('Manual review windows apply only to the inspected rollout, not new runs.')
        windows = REVIEW_WINDOWS[route]
        stable_stop = (item['teacher_decision'] == 'stop'
                       and item['teacher_decision_persistence_s'] >= .5)
        category, rationale = '标签不确定', '普通同行驶上下文，未证明教师决策更优。'
        if item['teacher_decision_persistence_s'] < .5:
            rationale = '教师决策类别持续不足0.5秒；暂不学习短暂停车或起步跳变。'
        elif item['teacher_decision'] == 'slow':
            rationale = '教师主要要求低速蠕行；不足以证明当前可以安全恢复行驶。'
        elif item['student_target_speed_mps'] < .5 and item['teacher_decision'] == 'go':
            rationale = '虽有起步分歧，但交叉交通或近距离障碍仍在；无法确认安全通行。'
        if stable_stop and in_windows(item['frame'], windows['better']) and item['student_target_speed_mps'] > 2:
            category = '教师决策更合理'
            rationale = ('前方车辆或交叉车辆造成冲突，教师持续要求停车，学生仍要求行驶。'
                         if route in ('2215', '24816') and item['frame'] > 30
                         else 'RGB显示停车标志路口/交叉交通或近距离前车；教师停车更早。')
        elif stable_stop and in_windows(item['frame'], windows['wait']) and item['student_target_speed_mps'] < .5:
            category = '合理等待'
            rationale = '路口仍有交叉交通或前车占据空间，双方持续要求停车；保留为等待对照。'
        if route == '2127' and item['frame'] == 45:
            category, rationale = '标签不确定', '路线结束帧出现显著轨迹分歧，教师停车仅持续0.2秒；不作为轨迹修正标签。'
        if item['measured_speed_mps'] < -.2:
            category, rationale = '标签不确定', '碰撞后的负向纵向速度/反弹状态；不用于正常起步或停车监督。'
        ego = item['ego']
        yaw = math.radians(ego['rotation_deg'][2])
        velocity = ego['velocity_mps']
        actual_forward = velocity[0] * math.cos(yaw) + velocity[1] * math.sin(yaw)
        reviewed.append(dict(item, review_category=category, review_rationale=rationale,
                             requires_behavioral_review=False,
                             actual_forward_speed_mps=actual_forward,
                             speed_estimate_error_mps=item['measured_speed_mps'] - actual_forward,
                             supervision_scope='target_speed_only' if category in ('教师决策更合理', '合理等待') else 'none',
                             trajectory_supervision_approved=False,
                             student_already_braking=item['student_control']['brake'] > .5,
                             review_method='RGB contact sheets + 20Hz decision persistence + actor geometry + trajectory overlays; no counterfactual rollout'))
    groups = {category: [r for r in reviewed if r['review_category'] == category]
              for category in ('教师决策更合理', '合理等待', '标签不确定')}
    for category, filename in [('教师决策更合理', 'speed_correction_frames.txt'),
                               ('合理等待', 'reasonable_wait_frames.txt'),
                               ('标签不确定', 'uncertain_frames.txt')]:
        (output / filename).write_text(''.join(r['key'] + '\n' for r in groups[category]))
    # A small contrast set avoids overwhelming rare corrections with repeated stops.
    anchors = []
    for route in REVIEW_WINDOWS:
        waits = [r for r in groups['合理等待'] if r['route_id'].endswith('_' + route)]
        blocks = []
        for row in waits:
            if not blocks or row['time_s'] - blocks[-1][-1]['time_s'] > .5:
                blocks.append([])
            blocks[-1].append(row)
        for block in blocks:
            anchors.extend(block[:1] if len(block) == 1 else [block[0], block[-1]])
    (output / 'wait_anchor_frames.txt').write_text(''.join(r['key'] + '\n' for r in anchors))
    with (output / 'reviewed_samples.jsonl').open('w') as handle:
        for row in reviewed:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    report = {'status': 'review_completed', 'candidate_frames': len(reviewed),
              'categories': {k: len(v) for k, v in groups.items()}, 'wait_anchors': len(anchors),
              'routes': {route: dict(Counter(r['review_category'] for r in reviewed
                                            if r['route_id'].endswith('_' + route))) for route in REVIEW_WINDOWS},
              'reliable_start_recovery_positive_frames': 0,
              'trajectory_supervision_approved': False,
              'scope': 'These manifests are reviewed speed-label candidates, not approval to train all native losses.',
              'limitations': ['No counterfactual driving: better decisions do not prove collision avoidance.',
                             'Nearby traffic-light actor states do not establish which signal governs ego.',
                             'Decision persistence is the full offline run length, including subsequent frames.',
                             'Instantaneous actor boxes do not predict future motion or establish legal right of way.',
                             'All five routes are development routes, not independent validation.'],
              'manual_windows': REVIEW_WINDOWS}
    report['correction_frames_with_student_already_braking'] = sum(
        r['student_already_braking'] for r in groups['教师决策更合理'])
    report['correction_frames_with_control_braking_difference'] = sum(
        not r['student_already_braking'] for r in groups['教师决策更合理'])
    report['native_manifest_validation'] = validate_manifests(output)
    (output / 'teacher_label_review_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    create_browser(reviewed, output)
    table = '\n'.join('| ' + route + ' | ' + ' | '.join(str(report['routes'][route].get(c, 0))
                       for c in groups) + ' |' for route in REVIEW_WINDOWS)
    (output / 'review_summary.md').write_text('''# 教师标签事件复盘

逐帧复盘282个候选样本，分类依据包括RGB、20Hz决策连续性、实际控制、周围车辆与双方轨迹。此为离线证据判断，不是教师实际接管后的安全验证。

| 路线 | 教师决策更合理 | 合理等待 | 标签不确定 |
| --- | ---: | ---: | ---: |
''' + table + '''

## 已确认的价值

有效样本主要支持提前停车/减速。部分帧学生实际已刹车，只是目标速度仍偏高；逐帧JSON记录了这一差别，不能把全部38帧描述为学生没有刹车。24816在6.05秒发生车辆碰撞，4.55–5.8秒窗口中有明确的教师停车要求；2215在178.50秒碰撞，177.55–178.05秒已有教师停车要求。只能认为这些要求更合理，不能断言足以避免碰撞。2204围栏碰撞发生在1.35秒，早于此次候选帧起点，路口停车标签不应被描述为该围栏碰撞的修复。

没有确认可直接用于解除停滞的正向起步样本。2204的186.05秒教师要求3.3m/s，但轨迹接近前车；197.30秒起步类别仅持续0.15秒。2215的181.30秒教师要求4.3m/s，前车间距约-0.4m，且起步类别仅持续0.25秒。上述样本排除。2127末帧的轨迹分歧也排除。2215的178.55秒负速度与真实纵向反弹方向一致，不能据此认定滤波器符号错误。

## 产物与下一步

`speed_correction_frames.txt`仅批准目标速度监督；`reasonable_wait_frames.txt`是等待对照，`wait_anchor_frames.txt`对连续等待取首尾，避免长停段重复占据训练；`uncertain_frames.txt`暂不学习。`reviewed_samples.jsonl`记录每帧证据与判断，`review.html`可按路线和结论浏览RGB、轨迹与车辆框。

下一步先接入按样本限制的速度损失，把修正样本和少量等待对照混入原数据，冻结骨干、小学习率微调速度头，控制新增数据占比；当前原生manifest加载器会同时提供轨迹和速度，因此不能直接把本清单投入所有训练损失。轨迹标签还需道路边界和导航方向核验。用相同种子复测开发路线并另设未参与采集的验证路线；同时比较碰撞、停滞时长、完成率，防止只学会更保守停车。

恢复起步应另行采集：先找前方可通行、无通行权冲突的稳定起步窗口，再短段教师接管检验，不能用本轮不确定的起步标签补足。
''')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def validate_manifests(output):
    import jsonpickle
    import jsonpickle.ext.numpy as jsonpickle_numpy
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '3rd_party/carla_garage/team_code'))
    from data import CARLA_Data
    jsonpickle_numpy.register_handlers()
    source = json.loads((output / 'review_source.json').read_text())
    settings = json.loads((Path(source['evaluations'][0]) / 'collection_config.json').read_text())
    checks = {}
    for filename in ('speed_correction_frames.txt', 'wait_anchor_frames.txt'):
        config = jsonpickle.decode((Path(settings['student_agent_dir']) / 'config.json').read_text())
        config.fmsr_recovery_root = source['fmsr_recovery_root']
        config.fmsr_recovery_manifest = str((output / filename).resolve())
        config.fmsr_frame_manifest = ''
        config.augment = config.use_color_aug = config.lidar_aug_prob = 0
        expected = (output / filename).read_text().splitlines()
        dataset = CARLA_Data(source['train_roots'], config)
        if len(dataset) != len(expected):
            raise ValueError(f'{filename}: selected {len(expected)}, loaded {len(dataset)}')
        for index in range(len(dataset)):
            sample = dataset[index]
            frame = int(dataset.sample_start[index])
            directory = Path(dataset.measurements[index][0].decode())
            with gzip.open(directory / f'{frame:04d}.json.gz', 'rt') as handle:
                label = json.load(handle)
            np.testing.assert_allclose(sample['target_speed_value'], label['target_speed'], rtol=1e-5)
            np.testing.assert_allclose(sample['route'][:config.fmsr_num_poses],
                                       label['teacher_checkpoints'], rtol=1e-5, atol=1e-5)
            if not np.isfinite(sample['rgb']).all() or not np.isfinite(sample['lidar']).all():
                raise ValueError('Nonfinite native sensor tensor')
        checks[filename] = {'status': 'passed', 'selected': len(expected), 'loaded': len(dataset),
                            'all_sensor_tensors_finite_and_teacher_labels_matched': True}
    return checks


def create_browser(reviewed, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.transforms import Affine2D
    assets = output / 'assets'
    assets.mkdir(exist_ok=True)
    cards = []
    for row in reviewed:
        stem = f"{row['route_id']}_{row['frame']:04d}"
        rgb = cv2.imread(row['rgb_path'])
        cv2.imwrite(str(assets / f'{stem}_rgb.jpg'), cv2.resize(rgb, (768, 320)))
        fig, ax = plt.subplots(figsize=(3.6, 3.6))
        navigation = np.asarray(row['navigation_route_ego_m'])
        mask = ((abs(navigation[:, 1]) < 25) & (navigation[:, 0] > -15) & (navigation[:, 0] < 40))
        # Separate route portions to avoid joining distant out-of-view vertices.
        navigation = navigation.copy()
        navigation[~mask] = np.nan
        ax.plot(navigation[:, 1], navigation[:, 0], '--', color='green', label='navigation')
        for key, color, name in [('student_trajectory_ego_m', 'blue', 'student'),
                                  ('teacher_trajectory_ego_m', 'orange', 'teacher')]:
            trajectory = np.asarray(row[key])
            ax.plot(trajectory[:, 1], trajectory[:, 0], '.-', color=color, label=name)
        for actor in row['nearby_dynamic_actors']:
            if abs(actor['x_m']) > 30 or abs(actor['y_m']) > 20:
                continue
            w, h = actor['extent'][1] * 2, actor['extent'][0] * 2
            box = Rectangle((-w / 2, -h / 2), w, h, color='red', alpha=.3)
            box.set_transform(Affine2D().rotate(-actor['yaw_rad']).translate(actor['y_m'], actor['x_m']) + ax.transData)
            ax.add_patch(box)
        ax.add_patch(Rectangle((-1, -2.4), 2, 4.8, color='gray', alpha=.5))
        ax.set(xlim=(-15, 15), ylim=(-8, 30), xlabel='ego right (m)', ylabel='ego forward (m)')
        ax.set_aspect('equal'); ax.grid(); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(assets / f'{stem}_bev.png', dpi=95); plt.close(fig)
        controls = html.escape(json.dumps({'student': row['student_control'], 'teacher': row['teacher_control']}))
        gap = f"{row['fmsr_actor']['front_gap_m']:.2f}m" if row['fmsr_actor'] else '无直线走廊前车'
        cards.append(f'''<article data-route="{row['route_id']}" data-category="{row['review_category']}">
<h3>{row['route_id']} f{row['frame']} / {row['time_s']:.2f}s — {row['review_category']}</h3>
<div class="pictures"><img src="assets/{stem}_rgb.jpg"><img src="assets/{stem}_bev.png"></div>
<p>{row['review_rationale']}</p><p>学生目标速度 {row['student_target_speed_mps']:.2f}；教师 {row['teacher_target_speed_mps']:.2f}；实际纵向 {row['actual_forward_speed_mps']:.2f} m/s；教师类别持续 {row['teacher_decision_persistence_s']:.2f}s；前车间距 {gap}；首6轨迹点平均分歧 {row['trajectory_first6_distance_m']:.2f}m；教师首6点到导航路线平均距离 {row['teacher_first6_navigation_distance_m']:.2f}m</p>
<details><summary>控制和样本索引</summary><pre>{controls}\n{html.escape(row['key'])}</pre></details></article>''')
    routes = ''.join(f'<option>{r}</option>' for r in sorted(set(r['route_id'] for r in reviewed)))
    categories = ''.join(f'<option>{c}</option>' for c in ('教师决策更合理', '合理等待', '标签不确定'))
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>教师标签事件复盘</title>
<style>body{font:16px sans-serif;margin:24px;background:#f4f5f7}article{background:white;padding:16px;margin:18px 0;max-width:1200px}.pictures{display:flex;gap:12px}.pictures img:first-child{width:65%;object-fit:contain}.pictures img:last-child{width:30%;object-fit:contain}pre{white-space:pre-wrap}select{padding:8px}nav{position:sticky;top:0;background:#f4f5f7;padding:12px}</style>
<h1>282个教师标签候选帧复盘</h1><p>蓝色：学生轨迹；橙色：教师轨迹；绿色虚线：导航路线；红框：当前周围车辆；灰框：自车。瞬时几何和导航距离不能证明未来安全或车道合规；此页未标注道路边界或主管信号。已批准样本仅用于目标速度监督。</p>
<nav>路线 <select id="route"><option value="">全部</option>''' + routes + '''</select> 结论 <select id="category"><option value="">全部</option>''' + categories + '''</select> <span id="count"></span></nav>''' + '\n'.join(cards) + '''<script>
function filter(){let n=0;const route=document.getElementById('route'),category=document.getElementById('category');document.querySelectorAll('article').forEach(a=>{let show=(!route.value||a.dataset.route===route.value)&&(!category.value||a.dataset.category===category.value);a.hidden=!show;n+=show});document.getElementById('count').textContent=n+' 帧'}
document.getElementById('route').onchange=filter;document.getElementById('category').onchange=filter;filter();
</script></html>'''
    (output / 'review.html').write_text(page)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    finalize(prepare(args.source, args.output), args.output)
