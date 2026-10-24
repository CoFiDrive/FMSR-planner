"""Opt-in observation wrapper; additional cameras are never policy inputs."""
import json
import os
from pathlib import Path

import cv2
import numpy as np
import carla
from functools import wraps
if os.environ.get('VISUAL_MODEL') == 'FMSR':
    from fmsrpdm import bootstrap
    bootstrap.configure_imports()
from sensor_agent import SensorAgent
if os.environ.get('ROBUSTNESS_SPEC'):
    from scripts.pdm_lite.robustness_agent import RobustnessAgent
    ObservationBase = RobustnessAgent
else:
    ObservationBase = SensorAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


def get_entry_point():
    return 'VisualCaseAgent'


class VisualCaseAgent(ObservationBase):
    def setup(self, *args, **kwargs):
        # Leaderboard may append successive route suffixes to agent_config.
        # Pass only the current (last) suffix so route-specific RNG/spec selection
        # never accidentally uses the first route in a multi-route evaluator.
        if args and '+' in str(args[0]):
            argument = str(args[0])
            args = (argument.split('+', 1)[0] + '+' + argument.rsplit('+', 1)[-1],) + args[1:]
        original_spec = os.environ.get('ROBUSTNESS_SPEC')
        if original_spec:
            manifest = json.loads(Path(original_spec).read_text())
            if 'route_conditions' in manifest:
                from scripts.pdm_lite.robustness_protocol import route_identity
                rid = route_identity(args[0]).split('_')[1]
                resolved = Path(original_spec).parent / 'resolved_specs' / (rid + '.json')
                resolved.parent.mkdir(exist_ok=True)
                resolved.write_text(json.dumps(dict(seed=manifest['seed'],
                    perturbation=manifest['route_conditions'][rid]), indent=2))
                os.environ['ROBUSTNESS_SPEC'] = str(resolved)
        try:
            super().setup(*args, **kwargs)
        finally:
            if original_spec:
                os.environ['ROBUSTNESS_SPEC'] = original_spec
        self.visual_tick = 0
        self.visual_file = None
        if os.environ.get('VISUAL_SAVE_INJECTION_ARRAYS') == '1' and hasattr(self, 'perturb'):
            self.capture_injection_arrays()

    def capture_injection_arrays(self):
        """Archive actual altered tensors; feature tensors are not road-space images."""
        directory = self.eval_metric_logger.directory / 'injection_arrays'
        directory.mkdir(exist_ok=True)
        self.injection_capture_count = 0
        self.injection_capture_ticks = set()
        def array(value):
            return value.detach().float().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)
        def instrument(name):
            original = getattr(self.perturb, name)
            @wraps(original)
            def capture(*args, **kwargs):
                before_count = len(self.perturb.events)
                result = original(*args, **kwargs)
                tick = self.perturb.tick
                if (len(self.perturb.events) > before_count
                        and tick not in self.injection_capture_ticks
                        and self.injection_capture_count < 40
                        and (self.injection_capture_count < 3 or tick % 100 == 0)):
                    index = self.injection_capture_count
                    self.injection_capture_count += 1
                    self.injection_capture_ticks.add(tick)
                    before = array(args[0])
                    after = array(result[0] if name == 'localization' else result)
                    stem = '{:03d}_tick{:05d}_{}'.format(index, tick, name)
                    np.savez_compressed(directory / (stem + '.npz'),
                                        before=before, after=after, delta=after-before)
                    (directory / (stem + '.json')).write_text(json.dumps(dict(
                        step=self.step, relative_time_s=self.perturb.time,
                        injection_method=name, before_shape=list(before.shape),
                        events=self.perturb.events[before_count:],
                        note='Actual input/output arrays. Feature channels are not road-space semantic labels.'), indent=2))
                return result
            setattr(self.perturb, name, capture)
        for name in ['state', 'feature', 'localization']:
            instrument(name)

    def sensors(self):
        original = super().sensors()
        self.visual_specs = original + [
            dict(type='sensor.camera.rgb', id='visual_third', x=-7., y=0., z=4.,
                 roll=0., pitch=-15., yaw=0., width=1280, height=720, fov=90),
            dict(type='sensor.camera.rgb', id='visual_top', x=0., y=0., z=35.,
                 roll=0., pitch=-90., yaw=0., width=1024, height=1024, fov=90)]
        return self.visual_specs

    def prepare_visual(self):
        logger = self.diagnostic_logger
        directory = logger.directory / 'visual'
        directory.mkdir(exist_ok=True)
        self.visual_directory = directory
        for name in ['front', 'third', 'top']:
            (directory / name).mkdir(exist_ok=True)
        (directory / 'camera_specs.json').write_text(json.dumps(self.visual_specs, indent=2))
        self.visual_file = (directory / 'frames.jsonl').open('x', buffering=1)
        road_map = CarlaDataProvider.get_world().get_map()
        maps = Path(os.environ['VISUAL_MAP_ROOT'])
        maps.mkdir(parents=True, exist_ok=True)
        town = road_map.name.rsplit('/', 1)[-1]
        if not (maps / (town + '_lanes.json')).exists():
            (maps / (town + '.xodr')).write_text(road_map.to_opendrive())
            lanes = {}
            for wp in road_map.generate_waypoints(2.):
                key = '{}_{}_{}'.format(wp.road_id, wp.section_id, wp.lane_id)
                tr = wp.transform
                right = tr.get_right_vector()
                loc = tr.location
                half = wp.lane_width / 2
                lanes.setdefault(key, []).append(dict(
                    s=wp.s, center=[loc.x, loc.y, loc.z],
                    left=[loc.x-right.x*half, loc.y-right.y*half],
                    right=[loc.x+right.x*half, loc.y+right.y*half],
                    junction=wp.is_junction,
                    left_marking=str(wp.left_lane_marking.type),
                    right_marking=str(wp.right_lane_marking.type)))
            for rows in lanes.values():
                rows.sort(key=lambda r: r['s'])
            (maps / (town + '_lanes.json')).write_text(json.dumps(dict(
                town=town, spacing_m=2., source='CARLA map.generate_waypoints', lanes=lanes)))
        self.visual_town = town

    def run_step(self, input_data, timestamp, sensors=None):
        # Only the original policy sensors reach the original agent.
        policy_input = {k: v for k, v in input_data.items() if not k.startswith('visual_')}
        control = super().run_step(policy_input, timestamp, sensors)
        if self.diagnostic_logger is not None:
            if self.visual_file is None:
                self.prepare_visual()
            if self.visual_tick % int(os.environ.get('VISUAL_EVERY', '5')) == 0:
                ego = CarlaDataProvider.get_hero_actor()
                transform = ego.get_transform()
                frame = CarlaDataProvider.get_world().get_snapshot().frame
                row = dict(simulation_frame=frame, agent_timestamp_s=float(timestamp),
                           town=self.visual_town, ego_matrix=transform.get_matrix(),
                           images={}, camera_world_matrices={}, sensor_frames={})
                for key, name in [('rgb_front', 'front'), ('visual_third', 'third'), ('visual_top', 'top')]:
                    if key not in input_data:
                        continue
                    relative = '{}/{:08d}.jpg'.format(name, frame)
                    if not cv2.imwrite(str(self.visual_directory / relative),
                                       input_data[key][1][:, :, :3], [cv2.IMWRITE_JPEG_QUALITY, 92]):
                        raise IOError(relative)
                    spec = next(s for s in self.visual_specs if s['id'] == key)
                    local = carla.Transform(carla.Location(x=spec['x'], y=spec['y'], z=spec['z']),
                                            carla.Rotation(pitch=spec['pitch'], yaw=spec['yaw'], roll=spec['roll']))
                    row['images'][name] = relative
                    row['sensor_frames'][name] = int(input_data[key][0])
                    row['camera_world_matrices'][name] = (np.asarray(transform.get_matrix()) @
                                                         np.asarray(local.get_matrix())).tolist()
                self.visual_file.write(json.dumps(row) + '\n')
        self.visual_tick += 1
        return control

    def destroy(self, results=None):
        if getattr(self, 'visual_file', None) is not None:
            self.visual_file.close()
        return super().destroy(results)
