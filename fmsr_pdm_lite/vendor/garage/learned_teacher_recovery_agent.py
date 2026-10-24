"""Student controls CARLA; frozen FMSR teacher labels the same visited states."""

import copy
import json
import os
from pathlib import Path

import carla
import numpy as np
import torch

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from sensor_agent import SensorAgent as PDMAgent
from fmsr.inference.sensor_agent_fmsr import SensorAgent as LEADAgent
from diagnostic_logger import DiagnosticLogger, diagnostic_directory, control_values, plain
from learned_teacher_labels import TeacherLabelWriter, validate_prediction, disagreement


def get_entry_point():
  return 'LearnedTeacherRecoveryAgent'


class CapturedStudent(PDMAgent):
  def _record_diagnostics(self, input_data, timestamp, tick_data, control, **policy):
    self.collection_tick = tick_data
    self.collection_policy = plain(policy)


class LearnedTeacherRecoveryAgent(AutonomousAgent):
  def setup(self, path_to_conf_file):
    self.student = self.teacher = self.writer = self.diagnostic_logger = None
    self._teacher_ready = self._student_ready = False
    self.step = -1
    self.stationary_ticks = 0
    self.latest_teacher_prediction = None
    self.track = Track.SENSORS
    settings = json.loads(Path(path_to_conf_file.split('+')[0]).read_text())
    self.settings = settings
    self.student = CapturedStudent.__new__(CapturedStudent)
    self.teacher = LEADAgent.__new__(LEADAgent)
    for agent in (self.student, self.teacher):
      agent._global_plan = self._global_plan
      agent._global_plan_world_coord = self._global_plan_world_coord
      agent.hero_actor = self.hero_actor
    student_argument = settings['student_agent_dir']
    # Saving belongs to the collection wrapper. Keep normal inference interfaces.
    saved_diagnostics = os.environ.pop('FMSR_DIAGNOSTICS', None)
    saved_path = os.environ.pop('SAVE_PATH', None)
    try:
      self.student.setup(student_argument)
      self._student_ready = True
    finally:
      if saved_path is not None:
        os.environ['SAVE_PATH'] = saved_path
      if saved_diagnostics is not None:
        os.environ['FMSR_DIAGNOSTICS'] = saved_diagnostics
    route_name = path_to_conf_file.rsplit('+', 1)[-1]
    self.teacher_log = Path(settings['collection_root']) / 'teacher_logs' / route_name
    self.teacher_log.mkdir(parents=True, exist_ok=False)
    original_path = os.environ.get('SAVE_PATH')
    os.environ['SAVE_PATH'] = str(self.teacher_log)
    try:
      self.teacher.setup(settings['teacher_agent_dir'] + '+' + route_name)
      self._teacher_ready = True
    finally:
      if original_path is None:
        os.environ.pop('SAVE_PATH', None)
      else:
        os.environ['SAVE_PATH'] = original_path
    # Capture numerical labels before teacher controller post-processing mutates them.
    teacher_forward = self.teacher.closed_loop_inference.forward
    def capture_forward(*args, **kwargs):
      prediction = teacher_forward(*args, **kwargs)
      self.latest_teacher_prediction = {
          'route': prediction.pred_route[0].detach().float().cpu().numpy().copy(),
          'speed': float(prediction.pred_target_speed_scalar.item()),
      }
      return prediction
    self.teacher.closed_loop_inference.forward = capture_forward
    self.student_specs = self.student.sensors()
    self.teacher_specs = self.teacher.sensors()
    shared = next(s for s in self.teacher_specs if s['id'] == 'lidar1')
    student_lidar = next(s for s in self.student_specs if s['id'] == 'lidar')
    if {k:v for k,v in shared.items() if k != 'id'} != {k:v for k,v in student_lidar.items() if k != 'id'}:
      raise ValueError('Teacher lidar1 and student LiDAR calibrations differ')
    if self.student.config.use_wp_gru or self.student.config.detect_boxes:
      raise ValueError('Collector supports the frozen PDM trajectory/speed task, not auxiliary targets')
    metadata = dict(settings, route_name=route_name, sample_every=self.student.config.data_save_freq,
                    shared_student_lidar='lidar1', student_sensor_specs=self.student_specs,
                    teacher_sensor_specs=self.teacher_specs)
    destination = Path(settings['collection_root']) / 'Recovery_Train' / (route_name.replace('_rep0_', '_Rep0_'))
    self.writer = TeacherLabelWriter(destination, metadata, self.student.config)
    directory = diagnostic_directory(settings['collection_root'], path_to_conf_file)
    self.diagnostic_logger = DiagnosticLogger(directory, metadata, sensor_every=0)
    print('Joint collection ready; student drives; teacher labels:', destination, flush=True)

  def sensors(self):
    # lidar1 exactly matches the PDM sensor; IMU/GPS/speed are identical too.
    sensors = copy.deepcopy(self.teacher_specs)
    existing = {s['id'] for s in sensors}
    for sensor in self.student_specs:
      if sensor['id'] == 'lidar':
        continue
      if sensor['id'] in existing:
        if next(s for s in sensors if s['id'] == sensor['id']) != sensor:
          raise ValueError(f'Shared sensor differs: {sensor["id"]}')
        continue
      sensors.append(copy.deepcopy(sensor))
    return sensors

  def set_scenario(self, scenario):
    self.teacher.set_scenario(scenario)

  @torch.inference_mode()
  def run_step(self, input_data, timestamp):
    self.step += 1
    raw_student = {s['id']: input_data['lidar1'] if s['id'] == 'lidar' else input_data[s['id']]
                   for s in self.student_specs}
    student_control = self.student.run_step(dict(raw_student), timestamp)
    teacher_control = self.teacher.run_step(dict(input_data), timestamp)
    # Both state estimators must use the action actually applied to the ego car.
    self.teacher.control = student_control
    tick = self.student.collection_tick
    policy = self.student.collection_policy
    speed = float(tick['speed'].item())
    self.stationary_ticks = self.stationary_ticks + 1 if speed < .2 else 0
    world = CarlaDataProvider.get_world()
    if self.step == 0:
      self.diagnostic_logger.bind(world, CarlaDataProvider.get_hero_actor())
    teacher = self.latest_teacher_prediction
    label_valid = teacher is not None and policy.get('phase') == 'inference'
    if label_valid:
      route, target_speed = validate_prediction(teacher['route'], teacher['speed'], self.student.config.fmsr_num_poses)
    else:
      route, target_speed = np.zeros((10, 2), dtype=np.float32), 0.
    student_speed = policy.get('control_target_speed_mps')
    reasons = disagreement(policy.get('predicted_trajectory_ego_m'), student_speed, route,
                            target_speed, speed, self.stationary_ticks) if label_valid else []
    self.diagnostic_logger.record(self.step, timestamp, raw_student, dict(policy,
        predicted_trajectory_ego_m=policy.get('predicted_trajectory_ego_m'),
        measured_speed_mps=speed, teacher_route_ego_m=route,
        teacher_target_speed_mps=target_speed, teacher_control=control_values(teacher_control),
        label_valid=label_valid, reasons=reasons), student_control, self.student.control)
    if self.step % self.student.config.data_save_freq == 0:
      # Save the full, aligned PDM LiDAR used for this observation, never teacher LiDAR preprocessing.
      lidar = self.student.lidar_buffer[-1] if len(self.student.lidar_buffer) else tick['lidar']
      command = self.student.commands[-2]
      measurement = {
          'label_source': 'learned_teacher', 'label_valid': label_valid,
          'teacher_checkpoints': route, 'route': route, 'target_speed': target_speed,
          'brake': int(target_speed < .01), 'steer': teacher_control.steer,
          'throttle': teacher_control.throttle, 'angle': 0.,
          'light_hazard': False, 'stop_sign_hazard': False, 'junction': False,
          'speed': speed, 'theta': tick['compass'], 'pos_global': tick['gps'],
          'command': command, 'next_command': command,
          'target_point': tick['target_point'].detach().cpu().numpy().reshape(2),
          'target_point_next': tick.get('target_point_next', tick['target_point']).detach().cpu().numpy().reshape(2),
          'augmentation_rotation': 0., 'augmentation_translation': 0.,
          'teacher_control': control_values(teacher_control), 'student_control': control_values(student_control),
          'sensor_frames': {name:int(value[0]) for name,value in input_data.items()},
      }
      self.writer.write(self.step // self.student.config.data_save_freq, world.get_snapshot().frame,
                        raw_student['rgb_front'][1], lidar, measurement,
                        {'reasons': reasons, 'label_valid': label_valid, 'speed': speed,
                         'teacher_target_speed': target_speed, 'student_target_speed': student_speed})
    return student_control

  def destroy(self, results=None):
    if self.writer is not None:
      self.writer.close()
    if self.diagnostic_logger is not None:
      self.diagnostic_logger.close()
    if self._teacher_ready:
      self.teacher.destroy()
      self._teacher_ready = False
    if self._student_ready:
      self.student.destroy()
      self._student_ready = False
