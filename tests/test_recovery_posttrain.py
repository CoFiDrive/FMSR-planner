"""Tests for recovery sampling, held-out-route exclusion and label selection."""

import argparse
import ast
import contextlib
import gzip
import importlib.util
import io
import json
import os
import pathlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from collections import Counter


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


recovery_data = load_module('recovery_data', ROOT / 'scripts/pdm_lite/recovery_data.py')
sampling = load_module('recovery_sampling', ROOT / '3rd_party/carla_garage/team_code/recovery_sampling.py')


class RecoveryPosttrainTests(unittest.TestCase):
    def test_recovery_continuation_argument_validation(self):
        # Exercise the entry-point checks without importing CARLA or starting CUDA.
        source = ROOT / '3rd_party/carla_garage/team_code/train.py'
        tree = ast.parse(source.read_text())
        main = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'main')
        start = next(i for i, node in enumerate(main.body)
                     if isinstance(node, ast.Assign) and any(
                         isinstance(target, ast.Attribute) and target.attr == 'logdir'
                         for target in node.targets))
        end = next(i for i in range(start, len(main.body))
                   if isinstance(main.body[i], ast.If) and
                   ast.unparse(main.body[i].test) == 'args.seed is not None')
        checks = compile(ast.Module(body=main.body[start:end], type_ignores=[]),
                         str(source), 'exec')
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            recovery = root / 'recovery'
            recovery.mkdir()
            base = dict(logdir=str(root), id='continued',
                        fmsr_recovery_root=str(recovery), use_fmsr=1,
                        trajectory_generator='residual_flow',
                        load_file=str(root / 'original/model_0002.pth'),
                        continue_epoch=1, flow_contract_enable=0,
                        fmsr_recovery_manifest=str(recovery / 'frames.txt'),
                        fmsr_recovery_fraction=.25, root_dir=[str(recovery)],
                        reset_scheduler=1)
            def validate(**overrides):
                args = SimpleNamespace(**(base | overrides))
                with contextlib.redirect_stderr(io.StringIO()):
                    exec(checks, dict(args=args, parser=argparse.ArgumentParser(),
                                      os=os, pathlib=pathlib))
            validate()
            validate(flow_contract_enable=1)
            validate(continue_epoch=0, reset_scheduler=0)
            for overrides in ({'load_file': None},
                              {'id': 'original'}, {'root_dir': []},
                              {'continue_epoch': 0}):
                with self.subTest(overrides=overrides), self.assertRaises(SystemExit):
                    validate(**overrides)

    def test_test_route_exclusion_by_id_and_location(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            def xml(name, x):
                path = root / name
                path.write_text(f'<routes><route id="1" town="Town12"><scenarios>'
                                f'<scenario><trigger_point x="{x}" y="0"/></scenario>'
                                f'</scenarios></route></routes>')
                return path
            test = xml('100.xml', 0)
            near = xml('200.xml', 10)
            far = xml('300.xml', 100)
            ids, points = recovery_data.test_signatures(test)
            self.assertTrue(recovery_data.overlaps_test(test, ids, points))
            self.assertTrue(recovery_data.overlaps_test(near, ids, points))
            self.assertFalse(recovery_data.overlaps_test(far, ids, points))

    def test_manifest_keeps_only_valid_pre_event_labels(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            route = root / 'Recovery_Train' / 'Town12_Rep0_sample'
            for sub in ('rgb', 'lidar', 'measurements'):
                (route / sub).mkdir(parents=True)
            for frame in range(10, 15):
                (route / 'rgb' / f'{frame:04d}.jpg').write_bytes(b'x')
                (route / 'lidar' / f'{frame:04d}.laz').write_bytes(b'x')
                label = {'route': [[float(i), 0.0] for i in range(20)],
                         'target_speed': 5.0}
                if frame == 12:
                    label['target_speed'] = float('nan')
                with gzip.open(route / 'measurements' / f'{frame:04d}.json.gz', 'wt') as handle:
                    json.dump(label, handle)
            rows = [{'frame': i, 'reasons': ['missed_hazard_braking'] if i == 14 else []}
                    for i in range(10, 15)]
            (route / 'recovery_events.jsonl').write_text(
                ''.join(json.dumps(row) + '\n' for row in rows))
            output = root / 'recovery_frames.txt'
            args = type('Args', (), {'recovery_root': root, 'history_frames': 3,
                                    'skip_first': 10, 'output': output})()
            recovery_data.build_manifest(args)
            self.assertEqual(output.read_text().splitlines(), [
                f'Recovery_Train/Town12_Rep0_sample/{i:04d}' for i in (11, 13, 14)])

    def test_distributed_sampler_keeps_global_recovery_ratio(self):
        flags = [False] * 80 + [True] * 20
        samplers = [sampling.RecoveryMixSampler(flags, .25, 2, rank, seed=9) for rank in (0, 1)]
        a, b = (list(s) for s in samplers)
        self.assertEqual(len(a), 50)
        self.assertEqual(len(b), 50)
        self.assertEqual(sum(flags[i] for i in a + b), 25)
        for sampler in samplers:
            sampler.set_epoch(1)
        self.assertNotEqual(a, list(samplers[0]))

    def test_small_recovery_set_caps_epoch_exposure(self):
        flags = [False] * 1000 + [True] * 5
        sampler = sampling.RecoveryMixSampler(flags, .25, 1, 0, max_recovery_repeats=4)
        drawn = list(sampler)
        self.assertEqual(len(drawn), 80)
        self.assertEqual(sum(flags[i] for i in drawn), 20)

    def test_scenario_balance_shortens_epoch_and_caps_each_frame(self):
        flags = [False] * 1000 + [True] * 110
        groups = {i: 'large' if i < 1100 else 'small' for i in range(1000, 1110)}
        sampler = sampling.RecoveryMixSampler(
            flags, .25, 1, 0, max_recovery_repeats=2,
            recovery_groups=groups, recovery_weights={i: 1. for i in groups},
            group_weights={'large': 1., 'small': 1.})
        indices = list(sampler)
        counts = Counter(i for i in indices if flags[i])
        self.assertEqual(len(indices), 160)
        self.assertEqual(Counter(groups[i] for i in indices if flags[i]), {'large': 20, 'small': 20})
        self.assertLessEqual(max(counts.values()), 2)
        self.assertEqual(len([i for i in counts if groups[i] == 'small']), 10)

    def test_long_stationary_weights_reduce_exposure_without_excluding_frames(self):
        flags = [False] * 1000 + [True] * 110
        groups = {i: 'large' if i < 1100 else 'small' for i in range(1000, 1110)}
        uniform = {i: 1. for i in groups}
        weighted = {i: .01 if 1000 <= i < 1080 else 1. for i in groups}
        def exposure(weights):
            sampler = sampling.RecoveryMixSampler(
                flags, .25, 1, 0, seed=3, max_recovery_repeats=2,
                recovery_groups=groups, recovery_weights=weights,
                group_weights={'large': 1., 'small': 1.})
            count = 0
            for epoch in range(20):
                sampler.set_epoch(epoch)
                count += sum(1000 <= i < 1080 for i in sampler)
            self.assertEqual(set(sampler.recovery_weights), set(groups))
            return count
        self.assertLess(exposure(weighted), exposure(uniform) / 4)

    def test_balanced_sampling_distributes_identical_global_plan_across_ranks(self):
        flags = [False] * 100 + [True] * 20
        groups = {i: 'priority' if i < 110 else 'other' for i in range(100, 120)}
        samplers = [sampling.RecoveryMixSampler(
            flags, .3, 2, rank, seed=9, max_recovery_repeats=4,
            recovery_groups=groups, recovery_weights={i: 1. for i in groups},
            group_weights={'priority': 2., 'other': 1.}) for rank in (0, 1)]
        combined = list(samplers[0]) + list(samplers[1])
        self.assertEqual(len(combined), 120)
        self.assertEqual(sum(flags[i] for i in combined), 36)
        self.assertEqual(Counter(groups[i] for i in combined if flags[i]), {'priority': 24, 'other': 12})
        self.assertEqual(samplers[0].last_epoch_stats, samplers[1].last_epoch_stats)

    def test_zero_frame_weights_are_rejected(self):
        with self.assertRaises(ValueError):
            sampling.RecoveryMixSampler([False, True], .5, 1, 0,
                                        recovery_groups={1: 'a'}, recovery_weights={1: 0.},
                                        group_weights={'a': 1.})


if __name__ == '__main__':
    unittest.main()
