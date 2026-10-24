"""Render four recorded cases using real CARLA map geometry and camera frames."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from PIL import Image

MODELS = ['FMSR', 'FM', 'FM_CR']
COLORS = {'FMSR': '#228833', 'FM': '#4477AA', 'FM_CR': '#CC6677'}


def records(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def project_path(policy, ego, camera, spec):
    xy = policy.get('predicted_trajectory_ego_m')
    if xy is None:
        return np.empty((0, 2))
    xy = np.asarray(xy)[:, :2]
    local = np.c_[xy, np.full(len(xy), .2), np.ones(len(xy))]
    world = local @ np.asarray(ego).T
    points = world @ np.linalg.inv(np.asarray(camera)).T
    # CARLA camera axes: x forward, y right, z up.
    forward = points[:, 0]
    focal = spec['width'] / (2 * np.tan(np.deg2rad(spec['fov']) / 2))
    result = np.c_[spec['width'] / 2 + focal * points[:, 1] / np.maximum(forward, .001),
                   spec['height'] / 2 - focal * points[:, 2] / np.maximum(forward, .001)]
    result[forward <= .1] = np.nan
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    args = parser.parse_args()
    base = args.campaign
    output = base / 'figures'
    output.mkdir(exist_ok=True)
    manifest = json.loads((base / 'manifest.json').read_text())
    route_ids = manifest['route_ids']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    fig, axes = plt.subplots(5, 4, figsize=(16, 17), constrained_layout=True)
    selection = {}
    for col, rid in enumerate(route_ids):
        cases = {}
        for model in MODELS:
            directories = sorted((base / model / 'diagnostics').glob('RouteScenario_' + rid + '_rep0*'))
            valid = [d for d in directories if (d / 'visual/frames.jsonl').exists()]
            if not valid:
                raise ValueError('No captured frames: {} {}'.format(model, rid))
            directory = valid[-1]
            frames = records(directory / 'frames.jsonl')
            visual = records(directory / 'visual/frames.jsonl')
            available = {r['simulation_frame']: r for r in frames}
            visual = [r for r in visual if r['simulation_frame'] in available
                      and available[r['simulation_frame']]['policy'].get('predicted_trajectory_ego_m')]
            if not visual:
                raise ValueError('No camera/plan alignment: {} {}'.format(model, rid))
            cases[model] = dict(directory=directory, frames=frames, visual=visual, available=available)
        reference = cases['FM_CR']
        # Fixed spatial reference at the middle of the driven route (by arc length).
        positions = np.asarray([r['ego']['position_m'][:2] for r in reference['frames']])
        progress = np.r_[0, np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))]
        center = positions[np.argmin(abs(progress - progress[-1] * .5))]
        for model, case in cases.items():
            selected = min(case['visual'], key=lambda r: np.linalg.norm(np.asarray(r['ego_matrix'])[:2, 3] - center))
            case['selected'] = selected
            case['row'] = case['available'][selected['simulation_frame']]
            case['spec'] = next(s for s in json.loads((case['directory'] / 'visual/camera_specs.json').read_text())
                                if s['id'] == 'visual_third')
            selection[model + '/' + rid] = dict(directory=str(case['directory']),
                simulation_frame=selected['simulation_frame'],
                distance_to_reference_m=float(np.linalg.norm(np.asarray(selected['ego_matrix'])[:2, 3] - center)))
        selected = reference['selected']
        context = Image.open(reference['directory'] / 'visual' / selected['images']['third'])
        axes[0, col].imshow(context)
        axes[0, col].set_title('Route ' + rid)
        context.save(output / ('route_' + rid + '_context.png'))
        Image.open(reference['directory'] / 'visual' / selected['images']['top']).save(
            output / ('route_' + rid + '_top.png'))
        map_ax = axes[1, col]
        town = selected['town']
        lanes = json.loads((base / 'maps' / (town + '_lanes.json')).read_text())['lanes']
        all_xy = np.concatenate([np.asarray([r['ego']['position_m'][:2] for r in c['frames']])
                                 for c in cases.values()])
        low, high = all_xy.min(0) - 12, all_xy.max(0) + 12
        map_ax.set_facecolor('#EEEEEE')
        for lane in lanes.values():
            xy = np.asarray([r['center'][:2] for r in lane])
            if len(xy) < 2 or np.any(xy.max(0) < low) or np.any(xy.min(0) > high):
                continue
            left = np.asarray([r['left'] for r in lane])
            right = np.asarray([r['right'] for r in lane])
            map_ax.add_patch(Polygon(np.r_[left, right[::-1]], facecolor='#555555', edgecolor='none'))
            for boundary in [left, right]:
                map_ax.plot(boundary[:, 0], boundary[:, 1], color='white', lw=.6)
        for model, case in cases.items():
            xy = np.asarray([r['ego']['position_m'][:2] for r in case['frames']])
            map_ax.plot(xy[:, 0], xy[:, 1], color=COLORS[model], lw=2,
                        label='FM+CR' if model == 'FM_CR' else model)
        map_ax.set(xlim=(low[0], high[0]), ylim=(high[1], low[1]), aspect='equal')
        map_ax.legend(loc='best', fontsize=8)
        for row_index, model in enumerate(MODELS, 2):
            case = cases[model]
            visual = case['selected']
            ax = axes[row_index, col]
            ax.imshow(Image.open(case['directory'] / 'visual' / visual['images']['third']))
            projected = project_path(case['row']['policy'], visual['ego_matrix'],
                                     visual['camera_world_matrices']['third'], case['spec'])
            if len(projected):
                ax.plot(projected[:, 0], projected[:, 1], '-o', color=COLORS[model], lw=2, ms=3)
            ax.set_xlim(0, case['spec']['width'])
            ax.set_ylim(case['spec']['height'], 0)
        for ax in axes[:, col]:
            ax.set_xticks([])
            ax.set_yticks([])
    for row, label in enumerate(['Context (FM+CR)', 'HD map / actual paths', 'FMSR', 'FM', 'FM+CR']):
        axes[row, 0].set_ylabel(label)
    for extension in ['png', 'pdf']:
        fig.savefig(output / ('four_case_comparison.' + extension), dpi=400, facecolor='white')
    plt.close(fig)
    (output / 'keyframe_selection.json').write_text(json.dumps(dict(
        protocol='nearest spatial position to midpoint of FM+CR driven arclength; inspect event timing manually',
        cameras='added observation sensors; not policy inputs', selection=selection), indent=2))


if __name__ == '__main__':
    main()
