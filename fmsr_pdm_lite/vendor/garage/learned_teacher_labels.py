"""Persist learned-teacher labels with the student's native PDM observations."""

import gzip
import json
from pathlib import Path

import cv2
import laspy
import numpy as np

from diagnostic_logger import plain


def validate_prediction(route, speed, count=10):
  route = np.asarray(route, dtype=np.float32)
  speed = float(speed)
  if route.shape != (count, 2) or not np.isfinite(route).all():
    raise ValueError(f'Invalid teacher route shape/values: {route.shape}')
  if not np.isfinite(speed) or not 0 <= speed <= 30:
    raise ValueError(f'Invalid teacher target speed: {speed}')
  return route, speed


def disagreement(student_route, student_speed, teacher_route, teacher_speed, speed, stationary_ticks):
  reasons = []
  if student_route is not None:
    route = np.asarray(student_route).reshape(-1, 2)
    teacher = np.asarray(teacher_route).reshape(-1, 2)
    count = min(len(route), len(teacher), 6)
    if np.linalg.norm(route[:count] - teacher[:count], axis=1).mean() > 2.5:
      reasons.append('plan_disagreement')
  if student_speed is not None:
    if abs(float(student_speed) - teacher_speed) > 2.0:
      reasons.append('speed_disagreement')
    if float(student_speed) < .5 and teacher_speed > 2.0:
      reasons.append('student_stop_teacher_go')
    if float(student_speed) > 2.0 and teacher_speed < .5:
      reasons.append('student_go_teacher_stop')
  if speed < .2 and teacher_speed > 2.0 and stationary_ticks >= 40:
    reasons.append('blocked_while_teacher_predicts_motion')
  return reasons


def verify_training_directory(directory, count=10):
  """Decode every saved observation and check same-frame teacher supervision."""
  directory = Path(directory)
  summary = json.loads((directory / 'collection_summary.json').read_text())
  total = summary['saved_frames']
  expected = {f'{frame:04d}' for frame in range(total)}
  for name, extension in [('rgb', '.jpg'), ('rgb_augmented', '.jpg'),
                          ('lidar', '.laz'), ('measurements', '.json.gz')]:
    actual = {p.name[:-len(extension)] for p in (directory / name).glob('*' + extension)}
    if actual != expected:
      raise ValueError(f'Incomplete {name} files in {directory}')
  events = [json.loads(line) for line in (directory / 'recovery_events.jsonl').read_text().splitlines()]
  if len(events) != total:
    raise ValueError('Recovery event/sample count differs')
  valid = 0
  previous = None
  for frame in range(total):
    stem = f'{frame:04d}'
    with gzip.open(directory / 'measurements' / f'{stem}.json.gz', 'rt') as handle:
      measurement = json.load(handle)
    simulation_frame = measurement['simulation_frame']
    if measurement['frame'] != frame or (previous is not None and simulation_frame <= previous):
      raise ValueError('Training frame order is invalid')
    sensor_frames = measurement['sensor_frames']
    if not sensor_frames or any(value != simulation_frame for value in sensor_frames.values()):
      raise ValueError(f'Teacher/student sensors are not synchronized at sample {frame}')
    if events[frame]['frame'] != frame or events[frame]['simulation_frame'] != simulation_frame:
      raise ValueError('Event and observation frame differ')
    if measurement['label_valid']:
      route, _ = validate_prediction(measurement['teacher_checkpoints'], measurement['target_speed'], count)
      np.testing.assert_allclose(measurement['route'], route, rtol=1e-5, atol=1e-5)
      valid += 1
    for name in ('rgb', 'rgb_augmented'):
      image = cv2.imread(str(directory / name / f'{stem}.jpg'))
      if image is None or image.size == 0:
        raise ValueError(f'Unreadable {name} at sample {frame}')
    cloud = laspy.read(directory / 'lidar' / f'{stem}.laz')
    if not np.isfinite(np.asarray(cloud.xyz)).all():
      raise ValueError(f'Nonfinite LiDAR at sample {frame}')
    previous = simulation_frame
  if valid != summary['valid_teacher_frames'] or previous != summary['last_simulation_frame']:
    raise ValueError('Training summary disagrees with saved files')
  return dict(directory=str(directory), saved_frames=total, valid_teacher_frames=valid,
              all_sensor_frames_synchronized=True, all_observations_decoded=True)


class TeacherLabelWriter:
  def __init__(self, directory, metadata, config):
    self.directory = Path(directory)
    self.directory.mkdir(parents=True, exist_ok=False)
    for name in ('rgb', 'rgb_augmented', 'lidar', 'measurements'):
      (self.directory / name).mkdir()
    self.config = config
    self.count = self.valid = 0
    self.events = (self.directory / 'recovery_events.jsonl').open('x', buffering=1)
    self.last_simulation_frame = None
    (self.directory / 'collection_metadata.json').write_text(json.dumps(plain(dict(metadata,
        schema_version=1, label_source='learned_teacher', controlled_by='student',
        route_representation='teacher_checkpoints_ego_m', trainable_as_benchmark_development_data=True)),
        indent=2) + '\n')

  def write(self, frame, simulation_frame, rgb_bgr, lidar_ego, measurement, event):
    if frame != self.count or (self.last_simulation_frame is not None and simulation_frame <= self.last_simulation_frame):
      raise ValueError('Training samples must be contiguous and simulation frames increasing')
    points = np.asarray(lidar_ego, dtype=np.float32)[:, :3]
    if points.ndim != 2 or not np.isfinite(points).all():
      raise ValueError('Invalid student LiDAR')
    if measurement['label_valid']:
      validate_prediction(measurement['teacher_checkpoints'], measurement['target_speed'],
                          self.config.fmsr_num_poses)
    stem = f'{frame:04d}'
    # Identity augmentation is explicit: camera geometry stays aligned to the labels.
    for name in ('rgb', 'rgb_augmented'):
      if not cv2.imwrite(str(self.directory / name / f'{stem}.jpg'), rgb_bgr[:, :, :3]):
        raise OSError('Failed to write student RGB')
    header = laspy.LasHeader(point_format=self.config.point_format)
    header.offsets = points.min(axis=0) if len(points) else np.zeros(3)
    header.scales = [self.config.point_precision] * 3
    with laspy.open(self.directory / 'lidar' / f'{stem}.laz', mode='w', header=header) as handle:
      record = laspy.ScaleAwarePointRecord.zeros(len(points), header=header)
      record.x, record.y, record.z = points[:, 0], points[:, 1], points[:, 2]
      handle.write_points(record)
    measurement = plain(dict(measurement, simulation_frame=simulation_frame, frame=frame))
    with gzip.open(self.directory / 'measurements' / f'{stem}.json.gz', 'wt') as handle:
      json.dump(measurement, handle, allow_nan=False)
    self.events.write(json.dumps(plain(dict(event, frame=frame, simulation_frame=simulation_frame)),
                                 allow_nan=False) + '\n')
    self.count += 1
    self.valid += int(measurement['label_valid'])
    self.last_simulation_frame = simulation_frame

  def close(self):
    if self.events.closed:
      return
    self.events.close()
    (self.directory / 'collection_summary.json').write_text(json.dumps({
        'saved_frames': self.count, 'valid_teacher_frames': self.valid,
        'last_simulation_frame': self.last_simulation_frame,
    }, indent=2) + '\n')
