"""Round-trip teacher labels through the native PDM training dataset."""

import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

import numpy as np

from config import GlobalConfig
from data import CARLA_Data
from learned_teacher_labels import TeacherLabelWriter, disagreement, validate_prediction, verify_training_directory


class LearnedTeacherCollectionTests(unittest.TestCase):
    def test_prediction_validation_and_directed_disagreements(self):
        route = np.column_stack([np.arange(10) + 2.5, np.ones(10)])
        validate_prediction(route, 5.)
        for trajectory, speed in [(route[:9], 5.), (route, float('nan')), (route, -1.)]:
            with self.assertRaises(ValueError):
                validate_prediction(trajectory, speed)
        reasons = disagreement(route, 0., route, 5., 0., 100)
        self.assertIn('student_stop_teacher_go', reasons)
        self.assertIn('blocked_while_teacher_predicts_motion', reasons)
        self.assertNotIn('plan_disagreement', reasons)

    def test_native_loader_preserves_teacher_checkpoints_and_speed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            directory = root / 'Recovery_Train/RouteScenario_2204_Rep0_Town12_Test'
            config = GlobalConfig()
            config.fmsr_num_poses = 10
            config.skip_first = 0
            config.use_wp_gru = 0
            config.use_plant = 0
            config.use_fmsr = 1
            config.detect_boxes = 0
            config.use_semantic = config.use_depth = config.use_bev_semantic = 0
            config.use_color_aug = config.augment = 0
            config.lidar_aug_prob = 0
            config.val_towns = []
            config.fmsr_recovery_root = str(root)
            config.fmsr_recovery_manifest = str(root / 'frames.txt')
            writer = TeacherLabelWriter(directory, {}, config)
            # Curved, nonuniform teacher samples would change under smoothing.
            route = np.column_stack([np.arange(10) * .7 + 1., np.arange(10) ** 2 * .04]).astype(np.float32)
            measurement = {
                'label_source': 'learned_teacher', 'label_valid': True,
                'teacher_checkpoints': route, 'route': route, 'target_speed': 5.25,
                'brake': 0, 'steer': .1, 'throttle': .2, 'angle': 0.,
                'light_hazard': False, 'stop_sign_hazard': False, 'junction': False,
                'speed': 3., 'theta': 0., 'pos_global': [1., 2.],
                'command': 4, 'next_command': 4, 'target_point': [10., 0.],
                'target_point_next': [20., 0.],
                'augmentation_rotation': 0., 'augmentation_translation': 0.,
            }
            points = np.array([[1., 2., -1.], [3., 4., 0.], [5., 6., 1.]], dtype=np.float32)
            image = np.zeros((512, 1024, 3), dtype=np.uint8)
            measurement['sensor_frames'] = {'rgb_front': 100, 'lidar1': 100}
            writer.write(0, 100, image, points, measurement, {'reasons': ['speed_disagreement']})
            measurement['sensor_frames'] = {'rgb_front': 105, 'lidar1': 105}
            writer.write(1, 105, image, points, measurement, {'reasons': []})
            with self.assertRaises(ValueError):
                writer.write(3, 110, image, points, measurement, {})
            writer.close()
            self.assertEqual(verify_training_directory(directory)['valid_teacher_frames'], 2)
            with gzip.open(directory / 'measurements/0001.json.gz', 'rt') as handle:
                broken = json.load(handle)
            broken['sensor_frames']['lidar1'] = 104
            with gzip.open(directory / 'measurements/0001.json.gz', 'wt') as handle:
                json.dump(broken, handle)
            with self.assertRaisesRegex(ValueError, 'not synchronized'):
                verify_training_directory(directory)
            (root / 'frames.txt').write_text(f'Recovery_Train/{directory.name}/0000\n')
            dataset = CARLA_Data([str(root / 'Recovery_Train')], config)
            self.assertEqual(len(dataset), 1)
            sample = dataset[0]
            np.testing.assert_allclose(sample['route'][:10], route)
            self.assertEqual(float(sample['target_speed_value']), 5.25)
            self.assertTrue(np.isfinite(sample['lidar']).all())
            with gzip.open(directory / 'measurements/0000.json.gz', 'rt') as handle:
                saved = json.load(handle)
            self.assertEqual(saved['simulation_frame'], 100)
            self.assertEqual(json.loads((directory / 'collection_summary.json').read_text())['saved_frames'], 2)


if __name__ == '__main__':
    unittest.main()
