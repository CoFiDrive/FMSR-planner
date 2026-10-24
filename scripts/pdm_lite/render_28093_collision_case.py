"""Event-anchored collision/no-collision figure from recorded route 28093."""
import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from PIL import Image
from render_visual_cases import records, project_path

MODELS = ['FMSR', 'FM', 'FM_CR']
LABELS = {'FMSR': 'FMSR', 'FM': 'FM', 'FM_CR': 'FM+CR'}
COLORS = {'FMSR': '#C76870', 'FM': '#4C8997', 'FM_CR': '#55A868'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, default=Path('/media/vcdc/T71/fmsr_visual_cases_20261021'))
    p.add_argument('--output', type=Path, default=Path('/home/vcdc/fmsr/outputs/analysis/visual_case_28093_collision_20261021'))
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    cases = {}
    events = []
    for model in MODELS:
        directory = next((args.campaign / model / 'diagnostics').glob('RouteScenario_28093_rep0*'))
        frames = records(directory / 'frames.jsonl')
        visual = records(directory / 'visual/frames.jsonl')
        collision = [r for r in records(directory / 'events.jsonl') if r['type'] == 'COLLISION_VEHICLE']
        events.extend(collision)
        scores = records_checkpoint(args.campaign / model / 'checkpoint_endpoint.json')
        cases[model] = dict(directory=directory, frames=frames, visual=visual,
                            indexed={r['simulation_frame']: r for r in frames}, collision=collision,
                            result=scores)
    conflict = np.mean([e['data']['location'][:2] for e in events], axis=0)
    manifest = dict(route='RouteScenario_28093_rep0', conflict_location_xy=conflict.tolist(), models={},
                    protocol='Collision cases: nearest recorded camera frame to first vehicle collision. '
                    'FM+CR: nearest ego position to mean collision location. Independent runs, not synchronized.',
                    map_source='CARLA Town04 OpenDRIVE and sampled actual lane boundaries',
                    image_cadence_seconds=.25)
    for model, case in cases.items():
        if case['collision']:
            event = case['collision'][0]
            target = event['event_frame']
            chosen = min(case['visual'], key=lambda r: abs(r['simulation_frame'] - target))
        else:
            event = None
            chosen = min(case['visual'], key=lambda r:
                         np.linalg.norm(np.asarray(r['ego_matrix'])[:2, 3] - conflict))
            target = chosen['simulation_frame']
        case.update(selected=chosen, target=target)
        case['row'] = case['indexed'][chosen['simulation_frame']]
        case['spec'] = next(s for s in json.loads((case['directory'] / 'visual/camera_specs.json').read_text())
                            if s['id'] == 'visual_third')
        case['elapsed'] = case['row']['world_timestamp_s'] - case['frames'][0]['world_timestamp_s']
        for view in ['front', 'third', 'top']:
            shutil.copy2(case['directory'] / 'visual' / chosen['images'][view],
                         out / (model + '_' + view + '_raw.jpg'))
        manifest['models'][model] = dict(source=str(case['directory']),
            selected_frame=chosen['simulation_frame'], sensor_frames=chosen['sensor_frames'],
            elapsed_seconds=case['elapsed'], collision_event=event,
            frame_difference_from_event=(chosen['simulation_frame'] - target) if event else None,
            scores=case['result']['scores'], status=case['result']['status'])

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(5, 1, figsize=(7.5, 18), constrained_layout=True)
    reference = cases['FM_CR']
    context = min(reference['visual'], key=lambda r: abs(r['simulation_frame'] - (reference['target'] - 40)))
    axes[0].imshow(Image.open(reference['directory'] / 'visual' / context['images']['third']))
    axes[0].set_title('Unsignalized left turn | Route 28093', fontsize=15)
    axes[0].set_ylabel('Scene context')
    map_ax = axes[1]
    paths = {m: np.asarray([r['ego']['position_m'][:2] for r in c['frames']]) for m, c in cases.items()}
    xy = np.concatenate(list(paths.values()))
    low, high = xy.min(0) - 8, xy.max(0) + 8
    # Fixed 16:9 map extent, retaining all driven trajectories.
    center = (low + high) / 2
    span = high - low
    width = max(span[0], span[1] * 16 / 9)
    low, high = center - [width / 2, width * 9 / 32], center + [width / 2, width * 9 / 32]
    lanes = json.loads((args.campaign / 'maps/Town04_lanes.json').read_text())['lanes']
    map_ax.set_facecolor('#EEEEEE')
    for lane in lanes.values():
        centers = np.asarray([r['center'][:2] for r in lane])
        if len(centers) < 2 or np.any(centers.max(0) < low) or np.any(centers.min(0) > high):
            continue
        left, right = np.asarray([r['left'] for r in lane]), np.asarray([r['right'] for r in lane])
        map_ax.add_patch(Polygon(np.r_[left, right[::-1]], facecolor='#555555', edgecolor='none'))
        for boundary in [left, right]:
            map_ax.plot(boundary[:, 0], boundary[:, 1], color='white', lw=.7)
    for model in MODELS:
        path = paths[model]
        map_ax.plot(path[:, 0], path[:, 1], color=COLORS[model], lw=2.3, label=LABELS[model])
        position = np.asarray(cases[model]['selected']['ego_matrix'])[:2, 3]
        map_ax.scatter(*position, color=COLORS[model], edgecolor='white', marker='^', s=65, zorder=8)
    map_ax.scatter(*conflict, marker='x', color='black', s=65, zorder=9)
    map_ax.set(xlim=(low[0], high[0]), ylim=(high[1], low[1]), aspect='equal', ylabel='HD map / actual paths')
    map_ax.legend(loc='lower left', fontsize=10, framealpha=.95)
    for i, model in enumerate(MODELS, 2):
        case = cases[model]
        ax = axes[i]
        show_camera(ax, case, case['selected'], overlay=True)
        ds = case['result']['scores']['score_composed']
        status = 'Collision' if case['collision'] else 'No collision recorded'
        ax.text(.02, .96, '{} | DS {:.0f}\n{} | t = {:.2f} s'.format(LABELS[model], ds, status, case['elapsed']),
                transform=ax.transAxes, va='top', fontsize=12, color='white',
                bbox=dict(facecolor='black', alpha=.7, edgecolor='none'))
        ax.set_ylabel(LABELS[model])
        # Separate annotated keyframe at native image aspect ratio.
        single, sax = plt.subplots(figsize=(8, 4.5))
        single.subplots_adjust(0, 0, 1, 1)
        show_camera(sax, case, case['selected'], overlay=True)
        sax.set_axis_off()
        single.savefig(out / (model + '_keyframe_trajectory.png'), dpi=300)
        plt.close(single)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    for extension in ['png', 'pdf']:
        fig.savefig(out / ('route28093_collision_comparison.' + extension), dpi=400, facecolor='white')
    plt.close(fig)

    sequence, saxes = plt.subplots(3, 3, figsize=(15, 9), constrained_layout=True)
    for row, model in enumerate(MODELS):
        case = cases[model]
        for col, offset in enumerate([-10, 0, 10]):
            visual = min(case['visual'], key=lambda r: abs(r['simulation_frame'] - (case['target'] + offset)))
            show_camera(saxes[row, col], case, visual, overlay=False)
            delta = (visual['simulation_frame'] - case['target']) * .05
            saxes[row, col].set_title('{} | reference {:+.2f} s'.format(LABELS[model], delta), fontsize=12)
            saxes[row, col].set_xticks([])
            saxes[row, col].set_yticks([])
    sequence.savefig(out / 'route28093_event_sequence.png', dpi=300)
    plt.close(sequence)
    (out / 'selection_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(out / 'route28093_collision_comparison.png')


def records_checkpoint(path):
    data = json.loads(path.read_text())
    return next(r for r in data['_checkpoint']['records'] if r['route_id'] == 'RouteScenario_28093_rep0')


def show_camera(ax, case, visual, overlay=False):
    ax.imshow(Image.open(case['directory'] / 'visual' / visual['images']['third']))
    if overlay:
        row = case['indexed'][visual['simulation_frame']]
        points = project_path(row['policy'], visual['ego_matrix'], visual['camera_world_matrices']['third'], case['spec'])
        if len(points):
            color = COLORS[next(m for m in MODELS if ('/' + m + '/') in str(case['directory']))]
            ax.plot(points[:, 0], points[:, 1], '-o', color=color, lw=2.5, ms=4)
    ax.set_xlim(0, case['spec']['width'])
    ax.set_ylim(case['spec']['height'], 0)


if __name__ == '__main__':
    main()
