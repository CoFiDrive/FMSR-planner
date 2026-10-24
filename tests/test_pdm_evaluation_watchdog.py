"""Process-exit/restart checks using tiny fake evaluations, never CARLA."""
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1] / 'scripts/pdm_lite/watch_evaluation.py'
spec = importlib.util.spec_from_file_location('watch_evaluation', MODULE)
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)


class WatchdogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.ids = ['RouteScenario_1_rep0', 'RouteScenario_2_rep0']

    def checkpoint(self, count, status='Started'):
        return {'entry_status': status, '_checkpoint': {
            'progress': [count, 2],
            'records': [{'route_id': x, 'status': 'Completed'} for x in self.ids[:count]],
            'global_record': {'status': 'Completed'} if status == 'Finished' else {}}}

    def fake_launcher(self, fail=False):
        fake = self.output / 'fake.py'
        fake.write_text('''import json, os, time
from pathlib import Path
p = Path(os.environ['EVALUATION_OUTPUT_DIR'])
attempts = p / 'attempts'
n = int(attempts.read_text()) + 1 if attempts.exists() else 1
attempts.write_text(str(n))
time.sleep(0.15)
''' + ('raise SystemExit(1)\n' if fail else '''
count = min(n, 2)
data = {'entry_status': 'Finished' if count == 2 else 'Started', '_checkpoint': {
    'progress': [count, 2],
    'records': [{'route_id': f'RouteScenario_{i+1}_rep0', 'status': 'Completed'} for i in range(count)],
    'global_record': {'status': 'Completed'} if count == 2 else {}}}
t = p / 'checkpoint.tmp'
t.write_text(json.dumps(data))
t.replace(p / 'checkpoint_endpoint.json')
raise SystemExit(0 if count == 2 else 1)
'''))
        launcher = self.output / 'fake.sh'
        launcher.write_text('#!/bin/bash\n' + shlex.quote(sys.executable) + ' ' + shlex.quote(str(fake)) + '\n')
        return launcher

    def monitor(self, launcher):
        return watch.Watchdog(self.output, self.ids, poll=.03, retry_delay=.03,
                              max_stalled=2, launcher=launcher)

    def test_crash_restarts_and_stops_at_complete(self):
        monitor = self.monitor(self.fake_launcher())
        self.assertEqual(monitor.run(), 0)
        self.assertEqual((self.output / 'attempts').read_text(), '2')
        self.assertEqual(json.loads((self.output / 'watchdog_status.json').read_text())['state'], 'completed')

    def test_cli_uses_requested_96_routes_for_validation_and_launch(self):
        routes = self.output / 'failed96.xml'
        routes.write_text('<routes>' + ''.join(
            f'<route id="{i}"/>' for i in range(96)) + '</routes>')
        with patch.object(sys, 'argv', ['watch_evaluation.py', '--output-dir',
                                      str(self.output), '--routes', str(routes)]), \
                patch.dict(os.environ), patch.object(watch, 'Watchdog') as monitor:
            monitor.return_value.run.return_value = 0
            self.assertEqual(watch.main(), 0)
            self.assertEqual(monitor.call_args.args[1],
                             [f'RouteScenario_{i}_rep0' for i in range(96)])
            self.assertEqual(os.environ['ROUTES'], str(routes.resolve()))

    def test_attaches_without_duplicate_launch(self):
        launcher = self.fake_launcher()
        env = dict(os.environ, EVALUATION_OUTPUT_DIR=str(self.output))
        process = subprocess.Popen(['bash', str(launcher)], env=env)
        self.addCleanup(lambda: process.wait(timeout=5))
        deadline = time.monotonic() + 3
        while not (self.output / 'attempts').exists() and time.monotonic() < deadline:
            time.sleep(.01)
        monitor = self.monitor(launcher)
        self.assertEqual(monitor.run(), 0)
        self.assertEqual(monitor.restart_count, 1)
        self.assertEqual((self.output / 'attempts').read_text(), '2')

    def test_repeated_no_progress_exits_stop(self):
        monitor = self.monitor(self.fake_launcher(fail=True))
        self.assertEqual(monitor.run(), 3)
        self.assertEqual((self.output / 'attempts').read_text(), '2')

    def test_partial_checkpoint_is_recovered_from_backup(self):
        monitor = self.monitor(self.fake_launcher())
        monitor.endpoint.write_text('{')
        watch.atomic_write(monitor.backup, self.checkpoint(0))
        self.assertEqual(monitor.run(), 0)
        self.assertEqual(len(list(self.output.glob('checkpoint_unreadable_*.json'))), 1)

    def test_unrecoverable_checkpoint_is_preserved(self):
        monitor = self.monitor(self.fake_launcher())
        monitor.endpoint.write_text('{')
        self.assertEqual(monitor.run(), 2)
        self.assertEqual(monitor.endpoint.read_text(), '{')
        self.assertFalse((self.output / 'attempts').exists())

    def test_only_matching_checkpoint_process_is_selected(self):
        table = {123: {'argv': ['python', 'evaluator.py', '--checkpoint=' + str(self.output / 'checkpoint_endpoint.json')], 'cwd': self.output},
                 456: {'argv': ['python', 'evaluator.py', '--checkpoint=/another/run/checkpoint_endpoint.json'], 'cwd': self.output}}
        self.assertEqual(watch.matching_processes(table, self.output, self.output / 'fake.sh'), {123})

    def test_orphan_cleanup_checks_pid_birth_time(self):
        monitor = self.monitor(self.output / 'fake.sh')
        monitor.carla = {123: 'old', 456: 'same'}
        with patch.object(watch, 'process_table', side_effect=[{123: {'birth': 'new'}, 456: {'birth': 'same'}}, {}]), \
                patch.object(watch.os, 'kill') as kill, patch.object(watch.time, 'sleep'):
            monitor.cleanup_orphans()
        kill.assert_called_once_with(456, watch.signal.SIGTERM)

    def test_missing_carla_requires_matching_pid_birth(self):
        monitor = self.monitor(self.output / 'fake.sh')
        monitor.carla = {123: 'original'}
        self.assertFalse(monitor.carla_alive({123: {'birth': 'reused'}}))
        self.assertTrue(monitor.carla_alive({123: {'birth': 'original'}}))

    def test_stalled_cleanup_only_targets_same_evaluation_process(self):
        monitor = self.monitor(self.output / 'fake.sh')
        table = {123: {'birth': 'original'}, 456: {'birth': 'other'}}
        with patch.object(watch, 'process_table', side_effect=[
                    {123: {'birth': 'original'}, 456: {'birth': 'other'}},
                    {123: {'birth': 'reused'}, 456: {'birth': 'other'}},
                ]), patch.object(watch.os, 'kill') as kill, \
                patch.object(watch.time, 'sleep'):
            monitor.terminate_stalled_evaluation({123}, table, 'test stall')
        kill.assert_called_once_with(123, watch.signal.SIGTERM)


if __name__ == '__main__':
    unittest.main()
