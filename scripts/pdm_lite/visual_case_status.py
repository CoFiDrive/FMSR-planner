"""Print progress and capture counts for the four-route visualization campaign."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--campaign', type=Path, default=Path('/media/vcdc/T71/fmsr_visual_cases_20261021'))
args = p.parse_args()
status = args.campaign / 'status.json'
print(status.read_text() if status.exists() else 'No campaign status yet')
total = 0
manifest = json.loads((args.campaign / 'manifest.json').read_text())
for model in manifest['models']:
    root = args.campaign / model
    checkpoint = root / 'checkpoint_endpoint.json'
    progress = [0, len(manifest['route_ids'])]
    if checkpoint.exists():
        try:
            d = json.loads(checkpoint.read_text())['_checkpoint']
            progress = d['progress']
            for record in d['records']:
                print(model, record['route_id'], record['status'],
                      'DS=', record['scores']['score_composed'])
        except (ValueError, KeyError):
            print('Checkpoint being updated:', checkpoint)
    total += progress[0]
    print(model, 'progress', progress)
    for directory in sorted((root / 'diagnostics').glob('*')):
        visual = directory / 'visual/frames.jsonl'
        if visual.exists():
            with visual.open() as f:
                n = sum(1 for _ in f)
            print(' ', directory.name, 'camera frame sets:', n)
print('Total route progress: {}/{}'.format(total, manifest['total_route_runs']))
