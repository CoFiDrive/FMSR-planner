"""Durable vehicle telemetry and Bench2Drive-compatible 10 Hz metric export."""
import json
from pathlib import Path
import time
import hashlib
import os
import inspect
import sys
from functools import lru_cache


@lru_cache(maxsize=32)
def file_hash(path, size, mtime_ns):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class EvalMetricLogger:
    def __init__(self, root, config_argument, route_index=None):
        argument = str(config_argument)
        # The appended name is the evaluator's record["save_name"].
        name = argument.rsplit('+', 1)[-1] if '+' in argument else str(route_index or '')
        if not name or name in ('.', '..') or Path(name).name != name:
            raise ValueError('Evaluation metrics require a valid +save_name or route_index')
        self.directory = Path(root) / 'eval_metrics' / name
        self.directory.mkdir(parents=True, exist_ok=True)
        # Preserve an earlier attempt if a caller explicitly reuses its name.
        names = ('vehicle_state.jsonl', 'planning.jsonl', 'metric_info.json', 'metadata.json', 'probes')
        if any((self.directory / n).exists() for n in names):
            archive = self.directory / 'attempts' / str(time.time_ns())
            archive.mkdir(parents=True)
            for n in names:
                if (self.directory / n).exists():
                    (self.directory / n).rename(archive / n)
        self.stream = (self.directory / 'vehicle_state.jsonl').open('x', buffering=1)
        self.planning = (self.directory / 'planning.jsonl').open('x', buffering=1)
        self.metrics = {}
        self.last_time = None
        self.last_sample = None
        self.last_frame = None
        self.count = 0
        self.metadata = {
            'schema_version': 1, 'save_name': name,
            'metric_sample_interval_s': 0.1,
            'metric_info_units': {'acceleration': 'm/s^2', 'angular_velocity': 'deg/s',
                                  'rotation': 'deg', 'location': 'm'},
            'metric_info_convention': 'raw CARLA values, matching Bench2Drive get_metric_info',
            'vehicle_state_cadence': 'every agent tick, including initialization and warmup',
            'control_semantics': 'actor applied control at capture, before current planning',
        }
        self._write_metadata()

    def _write_metadata(self):
        (self.directory / 'metadata.json').write_text(json.dumps(self.metadata, indent=2) + '\n')

    def set_run_metadata(self, config, nets, **settings):
        files = []
        code_paths = {str(Path(__file__).resolve()),
                      str(Path(__file__).with_name('planner_telemetry.py').resolve())}
        if settings.get('agent_file'):
            code_paths.add(str(Path(settings['agent_file']).resolve()))
        for net in nets:
            code_paths.add(inspect.getfile(type(net)))
            head = getattr(net, 'fmsr_head', getattr(net, 'planner', None))
            if head is not None:
                code_paths.add(inspect.getfile(type(head)))
                for method in ('flow_step', 'forward_test', 'sample'):
                    if hasattr(head, method):
                        code_paths.add(inspect.getfile(getattr(head, method)))
            path = Path(net._eval_checkpoint)
            stat = path.stat()
            config_path = path.parent / 'config.json'
            files.append({'checkpoint': str(path), 'sha256': file_hash(str(path), stat.st_size, stat.st_mtime_ns),
                          'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest()})
        fields = ('seed', 'trajectory_generator', 'fmsr_trajectory_head', 'flow_inference_steps',
                  'step_num', 'flow_contract_enable', 'flow_contract_method', 'flow_contract_rho',
                  'flow_contract_tau_kappa', 'fmsr_speed_source', 'fmsr_max_speed',
                  'inference_direct_controller')
        self.metadata.update(checkpoints=files, recording=settings,
                             runtime_config={k: getattr(config, k, None) for k in fields},
                             environment={k: os.environ.get(k) for k in
                                          ('SLOWER', 'STOP_CONTROL', 'DIRECT', 'TM_SEED', 'SEED')})
        self.metadata['code_sha256'] = {}
        for name in sorted(code_paths):
            stat = Path(name).stat()
            self.metadata['code_sha256'][name] = file_hash(name, stat.st_size, stat.st_mtime_ns)
        def cli_value(flag):
            for index, arg in enumerate(sys.argv[1:], 1):
                if arg.startswith(flag+'='):
                    return arg.split('=',1)[1]
                if arg == flag and index+1 < len(sys.argv):
                    return sys.argv[index+1]
            return None
        self.metadata['evaluator_cli'] = {key:cli_value('--'+key) for key in
                                          ('traffic-manager-seed','routes','repetitions')}
        routes = self.metadata['evaluator_cli']['routes']
        if routes and Path(routes).is_file():
            stat = Path(routes).stat()
            self.metadata['routes_sha256'] = file_hash(str(Path(routes).resolve()),stat.st_size,stat.st_mtime_ns)
        self._write_metadata()

    def save_probe(self, step, model_index, snapshot):
        import torch
        from scripts.pdm_lite.planner_telemetry import cpu_tree
        directory = self.directory / 'probes'
        directory.mkdir(exist_ok=True)
        relative = f'probes/{step:06d}_model{model_index}.pt'
        target = self.directory / relative
        temporary = target.with_suffix('.pt.tmp')
        torch.save(cpu_tree(dict(snapshot, step=step, simulation_frame=self.last_frame,
                                 timestamp_s=self.last_time)), temporary)
        temporary.replace(target)
        return relative

    def record(self, step, timestamp, metric, actor):
        snapshot = actor.get_world().get_snapshot()
        frame = int(snapshot.frame)
        now = float(snapshot.timestamp.elapsed_seconds)
        if self.last_frame is not None and (frame <= self.last_frame or now <= self.last_time):
            raise ValueError('Evaluation telemetry requires increasing simulation frames and time')
        velocity = actor.get_velocity()
        control = actor.get_control()
        row = dict(metric, step=int(step), simulation_frame=frame, timestamp_s=now,
                   agent_timestamp_s=float(timestamp),
                   velocity_mps=[float(velocity.x), float(velocity.y), float(velocity.z)],
                   applied_control={k: float(getattr(control, k)) for k in ('steer', 'throttle', 'brake')})
        self.stream.write(json.dumps(row, allow_nan=False) + '\n')
        self.count += 1
        # The official comfort script assumes dt=0.1, whereas CARLA ticks at 0.05.
        if self.last_sample is None or now - self.last_sample >= 0.1 - 1e-6:
            if self.last_sample is not None and abs(now - self.last_sample - 0.1) > 1e-5:
                self.metadata['irregular_metric_sampling'] = True
            self.metrics[str(step)] = metric
            self.last_sample = now
        self.last_time, self.last_frame = now, frame
        if self.count % 20 == 0:
            self.flush()

    def flush(self):
        target = self.directory / 'metric_info.json'
        temporary = target.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(self.metrics, allow_nan=False) + '\n')
        temporary.replace(target)
        self.metadata.update(vehicle_frame_count=self.count, metric_frame_count=len(self.metrics),
                             last_simulation_frame=self.last_frame)
        self._write_metadata()

    def record_plan(self, step, timestamp, policy, returned_control, filter_control):
        def plain(value):
            if hasattr(value, 'detach'):
                return plain(value.detach().cpu().tolist())
            if hasattr(value, 'tolist'):
                return plain(value.tolist())
            if isinstance(value, dict):
                return {str(k): plain(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [plain(v) for v in value]
            return value

        def controls(control):
            return {k: float(getattr(control, k)) for k in ('steer', 'throttle', 'brake')}

        row = {'step': int(step), 'simulation_frame': self.last_frame,
               'timestamp_s': self.last_time, 'agent_timestamp_s': float(timestamp),
               'policy': plain(policy), 'returned_control': controls(returned_control),
               'filter_control': controls(filter_control)}
        self.planning.write(json.dumps(row, allow_nan=False) + '\n')

    def close(self):
        if not self.stream.closed:
            self.flush()
            self.stream.close()
            self.planning.close()


def rebuild_metric_info(directory):
    """Recover the 10 Hz export from flushed telemetry after abrupt evaluator exit."""
    directory = Path(directory)
    metrics, last_sample = {}, None
    keys = ('acceleration', 'angular_velocity', 'forward_vector', 'right_vector', 'location', 'rotation')
    with (directory / 'vehicle_state.jsonl').open() as handle:
        for line in handle:
            row = json.loads(line)
            now = row['timestamp_s']
            if last_sample is None or now - last_sample >= 0.1 - 1e-6:
                metrics[str(row['step'])] = {key: row[key] for key in keys}
                last_sample = now
    target = directory / 'metric_info.json'
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(metrics, allow_nan=False) + '\n')
    temporary.replace(target)
    return len(metrics)
