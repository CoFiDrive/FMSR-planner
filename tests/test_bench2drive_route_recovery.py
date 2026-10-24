"""Regression checks for partial route cleanup and exact crash resume indices."""

import ast
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
LEADERBOARD = ROOT / '3rd_party/Bench2Drive/leaderboard/leaderboard'


def extract_class(path, name, namespace, methods=None):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    if methods is not None:
        node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def dictor(data, path):
    for part in path.split('.'):
        data = data.get(part) if isinstance(data, dict) else None
    return data


class RouteRecoveryTests(unittest.TestCase):
    def indexer(self):
        cls = extract_class(LEADERBOARD / 'utils/route_indexer.py', 'RouteIndexer', {
            'dictor': dictor, 'fetch_dict': lambda path: json.loads(Path(path).read_text())})
        indexer = cls.__new__(cls)
        indexer.total = 3
        indexer._configs_list = [NS(name=f'RouteScenario_{i}', repetition_index=0) for i in (1, 2, 3)]
        return indexer

    def resume(self, statuses, progress, entry='Crashed'):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'checkpoint.json'
            records = [{'route_id': f'RouteScenario_{i + 1}_rep0', 'status': status}
                       for i, status in enumerate(statuses)]
            path.write_text(json.dumps({'entry_status': entry, '_checkpoint': {
                'progress': [progress, 3], 'records': records}}))
            indexer = self.indexer()
            self.assertTrue(indexer.validate_and_resume(str(path)))
            return indexer.index

    def test_crashed_third_route_does_not_repeat_successful_second_route(self):
        self.assertEqual(self.resume(['Completed', 'Completed', 'Failed - Simulation crashed'], 3), 2)
        self.assertEqual(self.resume(['Completed', 'Completed', "Failed - Agent couldn't be set up"], 3), 2)

    def test_driving_failures_are_retained(self):
        self.assertEqual(self.resume(['Failed - TickRuntime', 'Failed - Agent got blocked'], 2, 'Started'), 2)

    def test_missing_record_retries_first_missing_route(self):
        self.assertEqual(self.resume(['Completed', 'Completed'], 3), 2)
        self.assertEqual(self.resume(['Completed'], 3), 1)

    def test_unidentified_crash_still_repeats_last_attempted_route(self):
        self.assertEqual(self.resume(['Completed', 'Completed'], 2), 1)

    def scenario_class(self):
        client = NS(apply_batch_sync=Mock())
        class Base:
            def _get_route(self, config):
                return []
            def _filter_scenarios(self, configurations):
                return configurations
            def _spawn_ego_vehicle(self):
                return None
            def remove_all_actors(self):
                self.base_cleanup_count = getattr(self, 'base_cleanup_count', 0) + 1
        cls = extract_class(LEADERBOARD / 'scenarios/route_scenario.py', 'RouteScenario', {
            'BasicScenario': Base,
            'CarlaDataProvider': NS(get_client=lambda: client, get_map=lambda: None),
            'carla': NS(command=NS(DestroyActor=lambda actor_id: ('destroy', actor_id)))},
            methods={'__init__', 'remove_all_actors', '__del__'})
        return cls, client

    def test_failed_ego_spawn_can_be_destroyed_without_missing_attributes(self):
        cls, client = self.scenario_class()
        instance = cls.__new__(cls)
        with self.assertRaisesRegex(ValueError, "couldn't spawn"):
            instance.__init__(None, NS(scenario_configs=[]))
        instance.__del__()
        self.assertEqual(instance._parked_ids, [])
        client.apply_batch_sync.assert_not_called()

    def test_explicit_cleanup_removes_parked_meshes_once(self):
        cls, client = self.scenario_class()
        instance = cls.__new__(cls)
        instance.client, instance._parked_ids = client, [100, 200]
        instance.remove_all_actors()
        instance.remove_all_actors()
        client.apply_batch_sync.assert_called_once_with([('destroy', 100), ('destroy', 200)])
        self.assertEqual(instance._parked_ids, [])

    def test_diagnostic_reload_cleans_same_town_without_changing_normal_default(self):
        for reload_setting, current_town in [('1', 'Town12'), ('0', 'Town12'), ('0', 'Town10')]:
            with self.subTest(reload=reload_setting, current_town=current_town):
                def world(town):
                    return NS(get_map=lambda: NS(name=town), get_settings=lambda: NS(),
                              apply_settings=Mock(), reset_all_traffic_lights=Mock(), tick=Mock())
                original, fresh = world(current_town), world('Town12')
                cls = extract_class(LEADERBOARD / 'leaderboard_evaluator_v2.py', 'LeaderboardEvaluator', {
                    'os': os, 'time': NS(time=lambda: 0),
                    'CarlaDataProvider': NS(set_client=Mock(), set_traffic_manager_port=Mock(),
                                           set_world=Mock(), get_map=lambda: NS(name='Town12'))},
                    methods={'_load_and_wait_for_world'})
                evaluator = cls.__new__(cls)
                evaluator.client = NS(get_world=lambda: original, load_world=Mock(return_value=fresh))
                evaluator.traffic_manager = NS(set_random_device_seed=Mock())
                with patch.dict(os.environ, {'CARLA_RELOAD_EACH_ROUTE': reload_setting}):
                    evaluator._load_and_wait_for_world(NS(traffic_manager_port=8010, traffic_manager_seed=0), 'Town12')
                should_reload = reload_setting == '1' or current_town != 'Town12'
                self.assertEqual(evaluator.client.load_world.called, should_reload)
                self.assertIs(evaluator.world, fresh if should_reload else original)
                evaluator.world.tick.assert_called_once()


if __name__ == '__main__':
    unittest.main()
