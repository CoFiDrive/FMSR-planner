"""Collect PDM-Lite supervision at states visited by a frozen student policy.

Run only on CARLA training routes with leaderboard_autopilot. The expert is
queried at every student state, but its control is never applied to the car.
"""

import json
import os
from pathlib import Path

import numpy as np

from data_agent import DataAgent
from sensor_agent import SensorAgent


def get_entry_point():
  return 'RecoveryAgent'


class RecoveryAgent(DataAgent):

  def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
    # The evaluator calls destroy even when setup raises halfway through.
    self._event_log = None
    self.student = None
    self._student_ready = False
    self._expert_ready = False
    super().setup(path_to_conf_file, route_index, traffic_manager)
    self._expert_ready = True
    student_dir = Path(os.environ['RECOVERY_STUDENT_AGENT_DIR']).resolve(strict=True)
    self.student = SensorAgent.__new__(SensorAgent)
    # The local evaluator supplies both a dense route and a downsampled GPS
    # route. SensorAgent consumes the latter two fields directly.
    self.student._global_plan = self._global_plan
    self.student._global_plan_world_coord = self._global_plan_world_coord
    saved_path = os.environ.pop('SAVE_PATH', None)
    try:
      self.student.setup(str(student_dir))
      self._student_ready = True
    except Exception:
      self.student = None
      raise
    finally:
      if saved_path is not None:
        os.environ['SAVE_PATH'] = saved_path
    self._expert_data = None
    self._slow_ticks = 0
    self._event_log = None
    if self.save_path is not None:
      self._event_log = (self.save_path / 'recovery_events.jsonl').open('a', encoding='utf-8')

  def sensors(self):
    specs = super().sensors()
    # Reuse DataAgent's RGB and LiDAR observations. SensorAgent additionally
    # needs a GNSS reading for its own route planner and UKF.
    specs.append({
        'type': 'sensor.other.gnss', 'x': 0.0, 'y': 0.0, 'z': 0.0,
        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'sensor_tick': 0.01,
        'id': 'gps'
    })
    return specs

  def save(self, *args, **kwargs):
    self._expert_data = super().save(*args, **kwargs)
    return self._expert_data

  def run_step(self, input_data, timestamp, sensors=None, plant=False):
    if plant:
      raise ValueError('RecoveryAgent must execute a student policy, not plant mode')
    # DataAgent rewrites input_data['lidar'] in place; keep the raw tuple for
    # SensorAgent so its normal preprocessing is applied exactly once.
    student_input = dict(input_data)
    student_input['rgb_front'] = student_input['rgb']
    expert_control = super().run_step(input_data, timestamp, sensors=sensors)
    student_control = self.student.run_step(student_input, timestamp)
    if self._event_log is not None and self.step % self.config.data_save_freq == 0:
      expert = self._expert_data or {}
      speed = float(expert.get('speed', 0.0))
      expert_speed = float(expert.get('target_speed', 0.0))
      if speed < 0.2 and expert_speed > 2.0:
        self._slow_ticks += self.config.data_save_freq
      else:
        self._slow_ticks = 0
      reasons = []
      if expert.get('vehicle_hazard') or expert.get('walker_hazard'):
        if student_control.brake < 0.5 and student_control.throttle > 0.1:
          reasons.append('missed_hazard_braking')
      if expert.get('light_hazard') and student_control.brake < 0.5:
        reasons.append('missed_red_light_braking')
      if student_control.brake > 0.8 and expert_speed > 3.0 and not expert_control.brake:
        reasons.append('unnecessary_braking')
      if self._slow_ticks >= 40:
        reasons.append('blocked_while_expert_can_move')
      if abs(student_control.steer - expert_control.steer) > 0.35 and speed > 2.0:
        reasons.append('steering_disagreement')
      predicted = getattr(self.student, 'last_pred_checkpoints', None)
      expert_route = np.asarray(expert.get('route', []), dtype=np.float32)
      if predicted is not None and expert_route.ndim == 2 and expert_route.shape[1] >= 2:
        steps = min(6, len(predicted), len(expert_route))
        if steps and float(np.linalg.norm(predicted[:steps, :2] - expert_route[:steps, :2],
                                       axis=1).mean()) > 2.5:
          reasons.append('plan_disagreement')
      route_index = self._waypoint_planner.route_index
      route_point = self._waypoint_planner.route_points[route_index, :2]
      ego_point = self._vehicle.get_location()
      route_error = float(np.linalg.norm(route_point - np.array([ego_point.x, ego_point.y])))
      if route_error > 2.5 and speed > 1.0:
        reasons.append('route_deviation')
      # Record all policy-induced states, including non-events, so the offline
      # selector can retain the 1-2 s window preceding a trigger.
      row = {
          'frame': self.step // self.config.data_save_freq,
          'reasons': reasons,
          'speed': speed,
          'expert_target_speed': expert_speed,
          'expert_brake': float(expert_control.brake),
          'student_brake': float(student_control.brake),
          'student_throttle': float(student_control.throttle),
          'route_error_m': route_error,
      }
      self._event_log.write(json.dumps(row) + '\n')
      self._event_log.flush()
    return student_control

  def destroy(self, results=None):
    if getattr(self, '_event_log', None) is not None:
      self._event_log.close()
      self._event_log = None
    try:
      if getattr(self, '_student_ready', False):
        self._student_ready = False
        self.student.destroy()
    finally:
      self.student = None
      if getattr(self, '_expert_ready', False):
        self._expert_ready = False
        super().destroy(results)
