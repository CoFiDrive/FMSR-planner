"""Check persisted telemetry, sensor alignment, terminal events and route selection."""

import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


logger_module = load('diagnostic_logger', ROOT / '3rd_party/carla_garage/team_code/diagnostic_logger.py')
selection = load('diagnostic_evaluation', ROOT / 'scripts/pdm_lite/diagnostic_evaluation.py')


class DiagnosticTests(unittest.TestCase):
    def make_world(self):
        position = NS(x=0., y=0., z=0.)
        rotation = NS(roll=0., pitch=0., yaw=0.)
        def actor(index, kind, location):
            return NS(id=index, type_id=kind, state='Green',
                      get_location=lambda: location,
                      get_transform=lambda: NS(location=location, rotation=rotation),
                      get_velocity=lambda: position,
                      bounding_box=NS(extent=position, location=position))
        ego = actor(1, 'vehicle.ego', position)
        pedestrian = actor(2, 'walker.pedestrian.0001', NS(x=2., y=0., z=0.))
        light = actor(3, 'traffic.traffic_light', NS(x=3., y=0., z=0.))
        far = actor(4, 'vehicle.far', NS(x=200., y=0., z=0.))
        self.frame = 100
        world = NS(get_snapshot=lambda: NS(frame=self.frame, timestamp=NS(elapsed_seconds=self.frame / 20)),
                   get_actors=lambda: [ego, pedestrian, light, far])
        return world, ego

    def test_streaming_survives_interruption_and_final_events_are_saved(self):
        with tempfile.TemporaryDirectory() as td:
            path = logger_module.diagnostic_directory(td, '/agent+old_route+current_route')
            world, ego = self.make_world()
            logger = logger_module.DiagnosticLogger(path, {}, sensor_every=2)
            logger.bind(world, ego)
            self.events = []
            criterion = NS(name='CollisionTest', events=self.events)
            logger.attach_scenario(NS(route=[], get_criteria=lambda: [criterion]))
            control = NS(steer=.1, throttle=.2, brake=0.)
            filter_control = NS(steer=0., throttle=0., brake=1.)
            inputs = {'rgb_front': (100, np.full((8, 8, 4), 128, dtype=np.uint8)),
                      'lidar': (99, np.array([[1., 2., 3., .5]], dtype=np.float32))}
            logger.record(0, 0., inputs, {'phase': 'initialization'}, control, filter_control)
            # The last frame must be readable before close, e.g. after simulator death.
            first = json.loads((path / 'frames.jsonl').read_text())
            self.assertEqual(first['sensor_frames']['lidar'], 99)
            self.assertNotEqual(first['returned_control'], first['filter_control'])
            self.assertEqual([x['id'] for x in first['nearby_actors']], [2, 3])
            self.frame += 1
            logger.record(1, .05, inputs, {'predicted_trajectory_ego_m': np.zeros((10, 2))}, control, control)
            # Criteria are ticked after the policy; a terminal collision can arrive at destroy.
            self.events.append(NS(get_type=lambda: NS(name='COLLISION_VEHICLE'),
                                  get_frame=lambda: 101, get_dict=lambda: {'other_actor': ego,
                                      'location': NS(x=1., y=2., z=3.)},
                                  get_message=lambda: 'Collision'))
            logger.close()
            logger.close()
            report = logger_module.verify_directory(path)
            self.assertEqual((report['frames'], report['predictions'], report['events']), (2, 1, 1))
            with np.load(path / first['sensor_files']['lidar']) as saved:
                np.testing.assert_array_equal(saved['points'], inputs['lidar'][1])
            with self.assertRaises(FileExistsError):
                logger_module.DiagnosticLogger(path, {})
            (path / first['sensor_files']['lidar']).unlink()
            with self.assertRaises(OSError):
                logger_module.verify_directory(path)

    def test_invalid_route_name_and_missing_name_are_rejected(self):
        for argument in ('/agent', '/agent+../bad', '/agent+..'):
            with self.subTest(argument=argument), self.assertRaises(ValueError):
                logger_module.diagnostic_directory('/out', argument)

    def test_selection_excludes_simulator_crashes_and_preserves_route_xml(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            def record(i, score, status='Completed', duration=10):
                return {'route_id': f'RouteScenario_{i}_rep0', 'scores': {'score_composed': score},
                        'status': status, 'meta': {'duration_game': duration}}
            baseline = [record(1, 100), record(2, 50), record(3, 100), record(4, 40)]
            current = [record(1, 50), record(2, 60), record(3, 0, 'Failed - Simulation crashed', 0),
                       record(4, 40, 'Failed - TickRuntime')]
            for name, records in [('old', baseline), ('new', current)]:
                (root / f'{name}.json').write_text(json.dumps({'_checkpoint': {'records': records, 'progress': [4, 4]}}))
            xml = root / 'all.xml'
            xml.write_text('<routes>' + ''.join(f'<route id="{i}" town="Town12"><weathers><weather sun_altitude_angle="42"/></weathers></route>' for i in range(1, 5)) + '</routes>')
            args = NS(baseline=root / 'old.json', current=root / 'new.json', routes=xml,
                      selection='union', route_id='', output=root / 'selected.xml', report=root / 'report.json')
            selection.select_routes(args)
            report = json.loads(args.report.read_text())
            self.assertEqual([r['route_id'] for r in report['routes']], ['RouteScenario_1_rep0', 'RouteScenario_4_rep0'])
            self.assertEqual(len(report['excluded_infrastructure_or_invalid_records']), 1)
            self.assertIn('sun_altitude_angle="42"', args.output.read_text())
            with self.assertRaises(FileExistsError):
                selection.select_routes(args)


if __name__ == '__main__':
    unittest.main()
