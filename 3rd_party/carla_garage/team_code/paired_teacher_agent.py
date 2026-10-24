"""Student rollout with synchronous learned-teacher labels (development routes)."""
import json
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch

from sensor_agent import SensorAgent as StudentAgent
from diagnostic_logger import plain, control_values
from fmsr.inference.sensor_agent_fmsr import SensorAgent as TeacherAgent


def get_entry_point():
    return 'PairedTeacherAgent'


def prediction_value(value):
    # NumPy cannot represent torch.bfloat16 used by the teacher's autocast.
    return plain(value.detach().float().cpu() if torch.is_tensor(value) else value)


class PairedTeacherAgent(StudentAgent):
    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        self.teacher = None
        self.pairs = None
        self.hook = None
        self.pair_count = 0
        super().setup(path_to_conf_file, route_index, traffic_manager)
        self.directory = self.diagnostic_logger.directory
        self.teacher = TeacherAgent.__new__(TeacherAgent)
        self.teacher._global_plan = self._global_plan
        self.teacher._global_plan_world_coord = self._global_plan_world_coord
        self.teacher_input_map = {}
        # Teacher inference must not change the student's stochastic flow sampler.
        numpy_state, python_state = np.random.get_state(), random.getstate()
        try:
            with torch.random.fork_rng(devices=[0]):
                self.teacher.setup(os.environ['PAIRED_TEACHER_DIR'])
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)
        self.student_inputs = None
        self.teacher_prediction = None
        original_student_forward = self.nets[0].forward

        def capture_student(*args, **kwargs):
            self._capture_student(None, args, kwargs)
            return original_student_forward(*args, **kwargs)

        # Garage invokes .forward directly, bypassing PyTorch module hooks.
        self.nets[0].forward = capture_student
        original_forward = self.teacher.closed_loop_inference.forward

        def capture_teacher(*args, **kwargs):
            prediction = original_forward(*args, **kwargs)
            self.teacher_prediction = {
                'trajectory_ego_m': prediction_value(prediction.pred_route),
                'target_speed_mps': prediction_value(prediction.pred_target_speed_scalar),
                'speed_distribution': prediction_value(prediction.pred_target_speed_distribution),
            }
            return prediction

        self.teacher.closed_loop_inference.forward = capture_teacher
        (self.directory / 'training').mkdir()
        (self.directory / 'teacher_sensors').mkdir()
        self.pairs = (self.directory / 'pairs.jsonl').open('x', buffering=1)
        self.sensor_specs = self.sensors()
        (self.directory / 'paired_metadata.json').write_text(json.dumps({
            'schema_version': 1, 'split': 'development',
            'teacher_dir': os.environ['PAIRED_TEACHER_DIR'],
            'teacher_checkpoint': str(next(Path(os.environ['PAIRED_TEACHER_DIR']).glob('model*.pth')).resolve()),
            'student_checkpoint': str(next(Path(self.config_path).glob('model*.pth')).resolve()),
            'driver': 'student', 'teacher_control_applied': False,
            'coordinates': 'ego x forward, y right; meters; target speed m/s',
            'student_inputs': 'exact tensors passed to student network, batch dimension removed',
            'sensors': self.sensor_specs, 'teacher_input_map': self.teacher_input_map,
        }, indent=2) + '\n')

    def _capture_student(self, module, args, kwargs):
        self.student_inputs = {
            key: value.detach().cpu().numpy()[0].copy()
            for key, value in kwargs.items() if torch.is_tensor(value)
        }

    def sensors(self):
        specs = super().sensors()
        for teacher_spec in self.teacher.sensors():
            signature = {k: v for k, v in teacher_spec.items() if k != 'id'}
            matched = next((s for s in specs if {k: v for k, v in s.items() if k != 'id'} == signature), None)
            if matched is None:
                matched = dict(teacher_spec, id='teacher_' + teacher_spec['id'])
                specs.append(matched)
            self.teacher_input_map[teacher_spec['id']] = matched['id']
        return specs

    def set_scenario(self, scenario):
        self.teacher.set_scenario(scenario)

    def run_step(self, input_data, timestamp, sensors=None):
        if self.step % 100 == 0 and shutil.disk_usage(self.directory).free < 3 * 1024**3:
            raise RuntimeError('Collection stopped: less than 3 GiB free disk space')
        self.student_inputs = self.teacher_prediction = None
        # Both agents may mutate observations during preprocessing.
        teacher_input = {key: (input_data[source][0],
            input_data[source][1].copy() if hasattr(input_data[source][1], 'copy') else input_data[source][1])
            for key, source in self.teacher_input_map.items()}
        student_control = super().run_step(dict(input_data), timestamp, sensors)
        numpy_state, python_state = np.random.get_state(), random.getstate()
        try:
            with torch.random.fork_rng(devices=[0]):
                teacher_control = self.teacher.run_step(teacher_input, timestamp)
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)
        # Any teacher state estimator must use the control actually applied.
        self.teacher.control = self.control
        snapshot = self.diagnostic_logger.world.get_snapshot()
        raw_files = {}
        for spec in self.sensor_specs:
            key = spec['id']
            if not key.startswith('teacher_'):
                continue
            frame, value = input_data[key]
            relative = f'teacher_sensors/{self.step:06d}_{key}'
            if spec['type'] == 'sensor.camera.rgb':
                relative += '.jpg'
                if not cv2.imwrite(str(self.directory / relative), value[:, :, :3], [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise IOError(relative)
            else:
                relative += '.npz'
                np.savez_compressed(self.directory / relative, value=value, sensor_frame=frame)
            raw_files[key] = relative
        sample = None
        if self.student_inputs is not None and self.teacher_prediction is not None:
            trajectory = np.asarray(self.teacher_prediction['trajectory_ego_m'], dtype=np.float32).reshape(-1, 2)
            speed = np.asarray(self.teacher_prediction['target_speed_mps'], dtype=np.float32).reshape(1)
            if len(trajectory) < 10 or not np.isfinite(trajectory).all() or not np.isfinite(speed).all():
                raise ValueError('Invalid synchronous teacher label')
            sample = f'training/{self.step:06d}.npz'
            np.savez_compressed(self.directory / sample, **self.student_inputs,
                teacher_trajectory=trajectory, teacher_target_speed=speed,
                simulation_frame=np.int64(snapshot.frame), step=np.int64(self.step))
            self.pair_count += 1
        self.pairs.write(json.dumps(plain({
            'step': self.step, 'simulation_frame': snapshot.frame, 'timestamp_s': timestamp,
            'sensor_frames': {key: value[0] for key, value in input_data.items()},
            'auxiliary_sensor_values': {key: value[1] for key, value in input_data.items()
                                       if key in ('gps', 'imu', 'speed')},
            'teacher': self.teacher_prediction, 'teacher_control': control_values(teacher_control),
            'applied_student_control': control_values(student_control),
            'teacher_sensor_files': raw_files, 'training_sample': sample,
        }), allow_nan=False) + '\n')
        if self.step % 100 == 0:
            print(f'Paired collection: step={self.step} training_samples={self.pair_count}', flush=True)
        return student_control

    def destroy(self, results=None):
        if self.pairs is not None:
            self.pairs.close()
            self.pairs = None
            (self.directory / 'paired_summary.json').write_text(json.dumps({'training_sample_count': self.pair_count}) + '\n')
            if self.pair_count:
                from scripts.pdm_lite.paired_data import verify
                verify(self.directory)
        if self.hook is not None:
            self.hook.remove()
            self.hook = None
        if self.teacher is not None:
            self.teacher.destroy()
            self.teacher = None
        if hasattr(self, 'nets') and hasattr(self, 'save_path'):
            super().destroy(results)
