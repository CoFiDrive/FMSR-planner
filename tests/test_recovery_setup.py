"""Regression checks for relocated disks and failed recovery initialization."""

import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


setup = load('recovery_setup', ROOT / 'scripts/pdm_lite/recovery_setup.py')
mount_data = load('mount_carla_garage_data', ROOT / 'scripts/pdm_lite/mount_carla_garage_data.py')


class RecoverySetupTests(unittest.TestCase):
    def test_compressed_dataset_view_preserves_raw_scenarios_and_mounts_read_only(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data_root = root / 'carla_garage_data'
            raw = data_root / 'noScenarios'
            raw.mkdir(parents=True)
            image_root = root / 'carla_garage_squashfs'
            image_root.mkdir()
            image = image_root / 'Accident.sqfs'
            image.touch()
            view = root / 'view'
            mounted = set()
            def mount(command, **kwargs):
                self.assertEqual(command[:3], ['squashfuse', '-o', 'ro'])
                self.assertEqual(Path(command[3]), image)
                mounted.add(Path(command[4]))
            with patch.object(mount_data.subprocess, 'run', side_effect=mount) as run, \
                    patch.object(mount_data.os.path, 'ismount', side_effect=lambda p: p in mounted):
                self.assertEqual(mount_data.dataset_view(data_root, view), view)
                self.assertEqual((view / 'noScenarios').resolve(), raw)
                mount_data.dataset_view(data_root, view)
                self.assertEqual(run.call_count, 1)

    def test_dataset_mount_refuses_to_hide_existing_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data_root = root / 'carla_garage_data'
            data_root.mkdir()
            image_root = root / 'carla_garage_squashfs'
            image_root.mkdir()
            (image_root / 'Accident.sqfs').touch()
            view = root / 'view'
            (view / 'Accident').mkdir(parents=True)
            existing = view / 'Accident' / 'data.txt'
            existing.write_text('retained data')
            with patch.object(mount_data.subprocess, 'run') as run:
                with self.assertRaisesRegex(ValueError, 'hide existing files'):
                    mount_data.dataset_view(data_root, view)
                run.assert_not_called()
            self.assertEqual(existing.read_text(), 'retained data')

    def assets(self, root):
        directory = root / 'media' / 'user' / 'T71' / 'pdm_lite_assets'
        directory.mkdir(parents=True)
        np.save(directory / 'anchors.npy', np.zeros((60, 11, 2)))
        np.savez(directory / 'stats.npz', mean=np.zeros((11, 2)), std=np.ones((11, 2)))
        config = {'py/object': 'config.GlobalConfig', 'fmsr_num_poses': 10,
                  'diffusion_speed': True, 'flow_inference_steps': 20,
                  'fmsr_anchor_path': str(root / 'gone' / 'anchors.npy'),
                  'fmsr_stats_path': str(root / 'gone' / 'stats.npz'),
                  'trajectory_stats_path': str(root / 'gone' / 'stats.npz')}
        checkpoint = root / 'checkpoint' / 'model_0011.pth'
        checkpoint.parent.mkdir()
        (checkpoint.parent / 'config.json').write_text(json.dumps(config))
        return directory, config, checkpoint

    def test_mount_relocation_updates_all_paths_preserves_source(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            directory, config, checkpoint = self.assets(root)
            setup.stage_config(checkpoint, root / 'agent', media_root=root / 'media')
            staged = json.loads((root / 'agent' / 'config.json').read_text())
            for key in setup.ASSET_FIELDS:
                self.assertEqual(Path(staged[key]).parent, directory)
            self.assertEqual(staged['flow_inference_steps'], 20)
            self.assertEqual(staged['py/object'], 'config.GlobalConfig')
            self.assertEqual(json.loads((checkpoint.parent / 'config.json').read_text()), config)

    def test_ambiguous_mounts_require_explicit_directory(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            directory, _, checkpoint = self.assets(root)
            other = directory.parent.parent / 'T72' / 'pdm_lite_assets'
            other.mkdir(parents=True)
            for asset in directory.iterdir():
                (other / asset.name).write_bytes(asset.read_bytes())
            with self.assertRaises(FileNotFoundError):
                setup.stage_config(checkpoint, root / 'agent', media_root=root / 'media')
            setup.stage_config(checkpoint, root / 'agent', asset_dir=directory)

    def test_invalid_stats_do_not_replace_staged_config(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            directory, _, checkpoint = self.assets(root)
            np.savez(directory / 'stats.npz', mean=np.zeros((11, 2)), std=np.zeros((11, 2)))
            destination = root / 'agent'
            destination.mkdir()
            (destination / 'config.json').write_text('original')
            with self.assertRaises(ValueError):
                setup.stage_config(checkpoint, destination, directory)
            self.assertEqual((destination / 'config.json').read_text(), 'original')

    def test_setup_failure_and_partial_results_are_retried(self):
        with tempfile.TemporaryDirectory() as td:
            result = Path(td) / 'result.json'
            data = {'entry_status': 'Finished', '_checkpoint': {
                'progress': [1, 1], 'records': [{'status': "Failed - Agent couldn't be set up",
                                              'meta': {'duration_game': 0}}]}}
            result.write_text(json.dumps(data))
            self.assertFalse(setup.route_complete(result))
            record = data['_checkpoint']['records'][0]
            record.update(status='Failed - Agent got blocked', meta={'duration_game': 90})
            result.write_text(json.dumps(data))
            self.assertTrue(setup.route_complete(result))
            for status in ('Perfect', 'Completed', 'Failed'):
                with self.subTest(status=status):
                    record['status'] = status
                    result.write_text(json.dumps(data))
                    self.assertTrue(setup.route_complete(result))
            for status in ('Failed - Agent crashed', 'Failed - Simulation crashed',
                           "Failed - Agent couldn't be set up", 'Started'):
                with self.subTest(status=status):
                    record['status'] = status
                    result.write_text(json.dumps(data))
                    self.assertFalse(setup.route_complete(result))
            data['_checkpoint']['progress'] = [0, 1]
            result.write_text(json.dumps(data))
            self.assertFalse(setup.route_complete(result))
            result.write_text('{')
            self.assertFalse(setup.route_complete(result))

    def test_student_setup_failure_cleanup_is_safe_and_restores_environment(self):
        calls = []

        class Expert:
            def setup(self, *args):
                self._global_plan = []
                self._global_plan_world_coord = []
                self.save_path = None

            def destroy(self, results=None):
                calls.append('expert')

        class Student:
            def setup(self, *args):
                raise FileNotFoundError('stats.npz')

            def destroy(self):
                raise AssertionError('Partial student must not use its full destructor')

        stubs = {'data_agent': types.SimpleNamespace(DataAgent=Expert),
                 'sensor_agent': types.SimpleNamespace(SensorAgent=Student)}
        with tempfile.TemporaryDirectory() as td, patch.dict('sys.modules', stubs), \
                patch.dict('os.environ', {'RECOVERY_STUDENT_AGENT_DIR': td, 'SAVE_PATH': td}):
            module = load('recovery_agent_test', ROOT / '3rd_party/carla_garage/team_code/recovery_agent.py')
            agent = module.RecoveryAgent()
            agent.destroy()  # Cleanup may even precede setup.
            with self.assertRaises(FileNotFoundError):
                agent.setup('route.xml')
            self.assertEqual(module.os.environ['SAVE_PATH'], td)
            agent.destroy()
            agent.destroy()
            self.assertEqual(calls, ['expert'])


if __name__ == '__main__':
    unittest.main()
