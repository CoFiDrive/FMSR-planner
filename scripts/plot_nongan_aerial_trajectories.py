"""Overlay recorded Nongan vehicle tracks on the original aerial photograph.

The aerial photograph has no georeferencing. Two visible anchors (the white ego
car and the roundabout centre) supply an approximate similarity transform;
vehicle motion and timestamps come only from traffic_state.jsonl.
"""

import argparse
import io
import json
import math
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/nongan_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import numpy as np
from PIL import Image


COLORS = {
    'ego_vehicle': '#cf3030',
    'inner1': '#007f5f',
    'inner2': '#e69f00',
    'outer1': '#8e44ad',
    'outer2': '#0096c7',
    'outer3': '#a98467',
    'outer4': '#7f7f00',
}
LABELS = {
    'ego_vehicle': 'Ego vehicle',
    'inner1': 'Inner 1',
    'inner2': 'Inner 2',
    'outer1': 'Outer 1',
    'outer2': 'Outer 2',
    'outer3': 'Outer 3',
    'outer4': 'Outer 4',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    photo_path = run / 'analysis_figures' / 'DJI_0133.JPG'
    image = Image.open(photo_path)
    width, height = image.size
    if (width, height) != (4032, 3024):
        raise ValueError(f'Calibration requires 4032x3024 photograph, got {image.size}')

    tracks = defaultdict(list)
    with (run / 'traffic_state.jsonl').open() as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            actors = ([row['ego']] if row.get('ego') else []) + row.get('actors', [])
            for actor in actors:
                name = actor['role_name'].split('/')[-1]
                if name in COLORS:
                    tracks[name].append((row['sim_time'], actor['x'], actor['y'], actor['theta']))
    if set(tracks) != set(COLORS):
        raise ValueError(f'Missing tracks: {set(COLORS) - set(tracks)}')

    ego_start = np.array(tracks['ego_vehicle'][0][1:3])
    heading = float(tracks['ego_vehicle'][0][3])
    # The manually inspected white ego car centre is at (2070, 2475) pixels.
    # The seven recorded roundabout tracks centre near (2052, 1530) pixels.
    ego_pixel = np.array([2070.0, 2475.0])
    pixels_per_metre = 9.5
    forward = np.array([math.cos(heading), math.sin(heading)])
    left = np.array([-math.sin(heading), math.cos(heading)])

    def project(world_xy):
        delta = np.asarray(world_xy) - ego_start
        return np.column_stack((
            ego_pixel[0] - pixels_per_metre * (delta @ left),
            ego_pixel[1] - pixels_per_metre * (delta @ forward),
        ))

    # Trim unused edges while retaining the roundabout, visible ego start,
    # and the northbound section of the recorded ego track.
    crop_left, crop_top, crop_right, crop_bottom = 220, 90, 3800, 2860
    crop_width = crop_right - crop_left
    crop_height = crop_bottom - crop_top
    plt.rcParams['font.family'] = 'Arial'
    output_dpi = 300
    font_scale = 200 / output_dpi  # Keep the same pixel canvas at higher print DPI.
    fig = plt.figure(figsize=(crop_width / output_dpi, crop_height / output_dpi), dpi=output_dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image, extent=(0, width, height, 0))
    handles = []
    for track_index, (name, color) in enumerate(COLORS.items()):
        samples = np.asarray(tracks[name])
        xy = project(samples[:, 1:3])
        visible = ((xy[:, 0] >= 0) & (xy[:, 0] <= width) &
                   (xy[:, 1] >= 0) & (xy[:, 1] <= height))
        shown = xy.copy()
        shown[~visible] = np.nan
        ax.plot(shown[:, 0], shown[:, 1], color='white', lw=8.5 * font_scale, alpha=.90,
                solid_capstyle='round', zorder=3)
        ax.plot(shown[:, 0], shown[:, 1], color=color, lw=5.0 * font_scale, alpha=.96,
                solid_capstyle='round', zorder=4)
        arrow_fractions = (.22, .66) if name == 'ego_vehicle' else (.11 + .105 * (track_index - 1),)
        for fraction in arrow_fractions:
            mid = int(fraction * (len(xy) - 1))
            before = max(0, mid - 10)
            after = min(len(xy) - 1, mid + 10)
            direction = xy[after] - xy[before]
            length = np.linalg.norm(direction)
            if visible[before] and visible[mid] and visible[after] and length > 4:
                direction /= length
                across = np.array([-direction[1], direction[0]])
                centre = xy[mid]
                arrow_length = 60 if name == 'ego_vehicle' else 54
                half_width = 20 if name == 'ego_vehicle' else 18
                tip = centre + direction * arrow_length * .55
                base = centre - direction * arrow_length * .45
                vertices = np.array([tip, base + across * half_width,
                                     base - across * half_width])
                ax.add_patch(Polygon(vertices, closed=True, facecolor=color,
                                     edgecolor='black', linewidth=.45 * font_scale,
                                     zorder=8))
        start = xy[0]
        if visible[0]:
            ax.scatter([start[0]], [start[1]], s=310 * font_scale**2, marker='^',
                       facecolor='white', edgecolor='black' if name == 'ego_vehicle' else color,
                       linewidth=3 * font_scale,
                       zorder=6)
        if visible[-1]:
            end = xy[-1]
            ax.scatter([end[0]], [end[1]], s=200 * font_scale**2, marker='o',
                       facecolor=color, edgecolor='white', linewidth=2 * font_scale,
                       zorder=7)
        handles.append(Line2D([0], [0], color=color, lw=5 * font_scale,
                              label=LABELS[name]))
    ax.set_xlim(crop_left, crop_right)
    ax.set_ylim(crop_bottom, crop_top)
    ax.axis('off')
    ax.text(ego_pixel[0] + 115, ego_pixel[1] + 5, 'Ego start',
            fontsize=31 * font_scale, fontweight='bold', color='black',
            va='center', ha='left', zorder=10,
            bbox=dict(boxstyle='round,pad=0.22', facecolor='white',
                      edgecolor=COLORS['ego_vehicle'], linewidth=2 * font_scale))
    legend = fig.legend(handles=handles, loc='upper left', ncol=1,
                        bbox_to_anchor=(.57, .93),
                        prop={'family': 'Arial', 'size': 37 * font_scale, 'weight': 'bold'},
                        frameon=True, fancybox=True, framealpha=.97, facecolor='white',
                        edgecolor='black', handlelength=1.25,
                        handletextpad=.30, labelspacing=.18, borderpad=.28)
    legend.get_frame().set_linewidth(1.4 * font_scale)
    output = run / 'analysis_figures' / 'DJI_0133_measured_trajectories.png'
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=output_dpi)
    plt.close(fig)
    buffer.seek(0)
    with Image.open(buffer) as rendered:
        rendered.save(output, format='PNG', optimize=True, dpi=(output_dpi, output_dpi))
        jpeg_output = output.with_suffix('.jpg')
        rendered.convert('RGB').save(jpeg_output, format='JPEG', quality=90,
                                     subsampling=0, optimize=True,
                                     progressive=True, dpi=(output_dpi, output_dpi))
    notes = dict(source_photo=str(photo_path), source_trajectories=str(run / 'traffic_state.jsonl'),
                 method='Similarity transform anchored to visible ego car and roundabout; approximate alignment',
                 ego_start_world_m=ego_start.tolist(), ego_start_pixel=ego_pixel.tolist(),
                 initial_heading_rad=heading, pixels_per_metre=pixels_per_metre,
                 crop_pixels=[crop_left, crop_top, crop_right, crop_bottom],
                 output_dpi=output_dpi, compact_jpeg=str(jpeg_output),
                 ego_endpoint_outside_photo=True,
                 caveat='No photo georeferencing is available; do not use overlay for distance measurements.')
    output.with_suffix('.json').write_text(json.dumps(notes, indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    main()
