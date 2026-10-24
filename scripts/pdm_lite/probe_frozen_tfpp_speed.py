"""Read-only same-frame speed decoding of the existing frozen TF++ branch."""
import json
from pathlib import Path
import sys
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.pdm_lite.bounded_fm_recovery import FLOW, config, native_flow_model, loader, inputs, write_json
from fmsr_pdm_lite.fmsrpdm.generation_validation import summarize
import torch


def main():
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.5)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    device = torch.device('cuda')
    root = ROOT/'outputs/analysis/bounded_speed_corrector_20261007'
    cfg = config(FLOW.parent/'config.json')
    model = native_flow_model(cfg).to(device).eval().requires_grad_(False)
    model.load_state_dict(torch.load(FLOW, map_location='cpu', weights_only=True), strict=True)
    old = ROOT/'outputs/analysis/bounded_fm_recovery_20261007'
    dataset, batches = loader(cfg, old/'validation_frames.txt', 4)
    rows = []
    for batch in batches:
        captured = []
        hook = model.target_speed_network.register_forward_hook(
            lambda _module, _inputs, output: captured.append(output.detach().cpu()))
        def no_flow(ego_query, *_args, **_kwargs):
            return ego_query.new_zeros(len(ego_query), 11, 2)
        try:
            with torch.no_grad(), patch.object(model.fmsr_head, 'forward_test_flow', side_effect=no_flow):
                model(**inputs(batch, device))
        finally:
            hook.remove()
        if len(captured) != 1:
            raise ValueError('Expected one frozen TF++ speed network call')
        rows.append(captured[0])
    frames = [str(Path(paths[0].decode()).resolve()/f'{dataset.sample_start[i]:04d}.json.gz')
              for i, paths in enumerate(dataset.measurements)]
    cache = torch.load(old/'flow_validation.pt', weights_only=True)
    if frames != cache['frames']:
        raise ValueError('TF++ speed probe loaded different frames')
    logits = torch.cat(rows); probability = logits.softmax(-1)
    centers = torch.tensor(cfg.target_speeds)
    threshold = float(cfg.brake_uncertainty_threshold)
    expected = (probability*centers).sum(-1)
    speeds = {'argmax': centers[logits.argmax(-1)], 'expected_speed': expected,
        'original_uncertainty_decoder': torch.where(probability[:, 0]>threshold, centers[0], expected)}
    generated = torch.load(root/'generated_validation.pt', weights_only=True)
    report = {'scope': '128 same development frames; frozen categorical speed branch; no training, labels only in metrics',
        'brake_probability_threshold': threshold, 'target_speed_centers': centers.tolist(), 'variants': {}}
    for name, speed in speeds.items():
        prediction = generated.clone(); prediction[:, -1, 0] = speed
        report['variants'][name] = {'mean': summarize(prediction, cache['target']),
            'predicted_speed_mps': speed.tolist()}
    write_json(root/'frozen_tfpp_speed_probe.json', report)
    print(json.dumps({name: row['mean'] for name, row in report['variants'].items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()
