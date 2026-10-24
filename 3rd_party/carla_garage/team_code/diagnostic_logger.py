"""Opt-in, streaming evaluation diagnostics. Never feeds data into the policy."""

import json
from pathlib import Path

import cv2
import numpy as np


def plain(value):
  if hasattr(value, 'detach'):
    value = value.detach().cpu().numpy()
  if isinstance(value, np.ndarray):
    return plain(value.tolist())
  if isinstance(value, np.generic):
    return plain(value.item())
  if isinstance(value, dict):
    return {str(k): plain(v) for k, v in value.items()}
  if isinstance(value, (tuple, list)):
    return [plain(v) for v in value]
  if isinstance(value, float) and not np.isfinite(value):
    return None
  if all(hasattr(value, key) for key in ('x', 'y', 'z')):
    return [float(value.x), float(value.y), float(value.z)]
  if hasattr(value, 'id') and hasattr(value, 'type_id'):
    return {'id': int(value.id), 'type': value.type_id}
  return value


def diagnostic_directory(root, config_argument, route_index=None):
  # Bench2Drive can append more than one suffix when reusing args.agent_config.
  name = str(route_index) if route_index is not None else str(config_argument).rsplit('+', 1)[-1]
  if route_index is None and '+' not in str(config_argument):
    raise ValueError('Diagnostics require a route_index or Bench2Drive +save_name')
  if not name or name in ('.', '..') or Path(name).name != name:
    raise ValueError(f'Invalid diagnostic route name: {name!r}')
  return Path(root) / 'diagnostics' / name


def control_values(control):
  return {key: float(getattr(control, key)) for key in ('steer', 'throttle', 'brake')}


def xyz(vector):
  return [float(vector.x), float(vector.y), float(vector.z)]


def actor_state(actor):
  transform = actor.get_transform()
  row = {'id': actor.id, 'type': actor.type_id, 'position_m': xyz(transform.location),
         'rotation_deg': [float(transform.rotation.roll), float(transform.rotation.pitch),
                          float(transform.rotation.yaw)]}
  if actor.type_id.startswith(('vehicle.', 'walker.')):
    row['velocity_mps'] = xyz(actor.get_velocity())
    row['bbox_extent_m'] = xyz(actor.bounding_box.extent)
    row['bbox_offset_m'] = xyz(actor.bounding_box.location)
  if actor.type_id.startswith('traffic.traffic_light'):
    row['traffic_light_state'] = str(actor.state)
  return row


class DiagnosticLogger:
  def __init__(self, directory, metadata, sensor_every=5, actor_radius=50.0):
    if sensor_every < 0 or actor_radius <= 0:
      raise ValueError('sensor_every must be >= 0 and actor_radius must be > 0')
    self.directory = Path(directory)
    # Refuse to overwrite a previous replay, including an interrupted one.
    self.directory.mkdir(parents=True, exist_ok=False)
    self.sensor_every = sensor_every
    self.actor_radius = actor_radius
    self.metadata = dict(metadata, schema_version=1, sensor_every=sensor_every,
                         actor_radius_m=actor_radius, route_name=self.directory.name,
                         sensor_coordinates='raw CARLA sensor coordinates; RGB saved as BGR JPEG')
    (self.directory / 'metadata.json').write_text(json.dumps(plain(self.metadata), indent=2) + '\n')
    if sensor_every:
      (self.directory / 'rgb').mkdir()
      (self.directory / 'lidar').mkdir()
    self.frames = (self.directory / 'frames.jsonl').open('x', buffering=1)
    self.events = (self.directory / 'events.jsonl').open('x', buffering=1)
    self.world = self.ego = self.scenario = None
    self._seen_events = {}
    self.count = self.sensor_count = self.event_count = self.prediction_count = 0
    self.first_frame = self.last_frame = None
    self.last_step = None

  def bind(self, world, ego):
    self.world, self.ego = world, ego

  def attach_scenario(self, scenario):
    self.scenario = scenario
    route = getattr(scenario, 'route', [])
    self.metadata['route_world_m'] = [xyz(transform.location) for transform, _ in route]
    (self.directory / 'metadata.json').write_text(json.dumps(plain(self.metadata), indent=2) + '\n')

  def capture_events(self, observed_frame):
    completion = None
    if self.scenario is None:
      return completion
    for criterion in self.scenario.get_criteria():
      for event in getattr(criterion, 'events', []):
        kind = event.get_type().name
        data = event.get_dict()
        if kind == 'ROUTE_COMPLETION':
          completion = (data or {}).get('route_completed')
          continue
        if kind == 'ROUTE_COMPLETED':
          completion = 100.0
        row = plain({'type': kind, 'criterion': criterion.name,
                     'event_frame': event.get_frame(), 'data': data,
                     'message': event.get_message()})
        signature = json.dumps(row, sort_keys=True, allow_nan=False)
        key = (id(criterion), id(event))
        if self._seen_events.get(key) != signature:
          self._seen_events[key] = signature
          row['observed_frame'] = observed_frame
          self.events.write(json.dumps(row, allow_nan=False) + '\n')
          self.event_count += 1
    return completion

  def record(self, step, timestamp, input_data, policy, returned_control, filter_control):
    snapshot = self.world.get_snapshot()
    frame = int(snapshot.frame)
    if self.last_step is not None and (step != self.last_step + 1 or frame <= self.last_frame):
      raise ValueError('Diagnostic frames must be consecutive agent steps and increasing simulation frames')
    ego = actor_state(self.ego)
    nearby, actor_errors = [], []
    ego_position = np.array(ego['position_m'])
    for actor in self.world.get_actors():
      if actor.id == self.ego.id or not actor.type_id.startswith(('vehicle.', 'walker.', 'traffic.traffic_light')):
        continue
      try:
        if np.linalg.norm(np.array(xyz(actor.get_location())) - ego_position) <= self.actor_radius:
          nearby.append(actor_state(actor))
      except RuntimeError as error:
        # Background actors can be removed between get_actors and get_transform.
        actor_errors.append({'id': actor.id, 'error': str(error)})
    sensors = {name: int(value[0]) for name, value in input_data.items()}
    files = {}
    if self.sensor_every and step % self.sensor_every == 0:
      stem = f'{frame:08d}'
      if 'rgb_front' in input_data:
        relative = f'rgb/{stem}.jpg'
        if not cv2.imwrite(str(self.directory / relative), input_data['rgb_front'][1][:, :, :3]):
          raise OSError(f'Could not save {relative}')
        files['rgb'] = relative
      if 'lidar' in input_data:
        relative = f'lidar/{stem}.npz'
        np.savez_compressed(self.directory / relative, points=input_data['lidar'][1],
                            sensor_frame=input_data['lidar'][0])
        files['lidar'] = relative
      self.sensor_count += 1
    row = plain({'step': step, 'simulation_frame': frame, 'agent_timestamp_s': timestamp,
                 'world_timestamp_s': snapshot.timestamp.elapsed_seconds,
                 'sensor_frames': sensors, 'sensor_files': files, 'ego': ego,
                 'nearby_actors': nearby, 'actor_capture_errors': actor_errors,
                 'route_completion_percent': self.capture_events(frame),
                 'policy': policy, 'returned_control': control_values(returned_control),
                 'filter_control': control_values(filter_control)})
    self.frames.write(json.dumps(row, allow_nan=False) + '\n')
    self.count += 1
    self.prediction_count += int(policy.get('predicted_trajectory_ego_m') is not None)
    self.first_frame = frame if self.first_frame is None else self.first_frame
    self.last_frame, self.last_step = frame, step

  def close(self):
    if self.frames.closed:
      return
    event_error = None
    try:
      self.capture_events(self.last_frame)
    except RuntimeError as error:
      event_error = str(error)  # CARLA may have died; retain all flushed records.
    finally:
      self.frames.close()
      self.events.close()
    (self.directory / 'summary.json').write_text(json.dumps({
        'frame_count': self.count, 'prediction_count': self.prediction_count,
        'sensor_sample_count': self.sensor_count, 'event_record_count': self.event_count,
        'first_frame': self.first_frame, 'last_frame': self.last_frame,
        'final_event_capture_error': event_error,
    }, indent=2) + '\n')


def verify_directory(directory, require_prediction=True):
  """Read back telemetry and decode saved sensors; raise on missing/corrupt data."""
  directory = Path(directory)
  metadata = json.loads((directory / 'metadata.json').read_text())
  summary = json.loads((directory / 'summary.json').read_text())
  count = predictions = samples = 0
  last_frame = last_step = None
  with (directory / 'frames.jsonl').open() as handle:
    for line in handle:
      row = json.loads(line)
      frame, step = row['simulation_frame'], row['step']
      if last_frame is not None and (frame <= last_frame or step != last_step + 1):
        raise ValueError(f'Non-monotonic frames in {directory}')
      for key in ('ego', 'nearby_actors', 'sensor_frames', 'policy', 'returned_control', 'filter_control'):
        if key not in row:
          raise ValueError(f'Missing field {key}')
      predictions += int(row['policy'].get('predicted_trajectory_ego_m') is not None)
      if metadata['sensor_every'] and step % metadata['sensor_every'] == 0:
        samples += 1
        for sensor, key in (('rgb_front', 'rgb'), ('lidar', 'lidar')):
          if sensor not in row['sensor_frames']:
            continue
          path = directory / row['sensor_files'][key]
          if key == 'rgb':
            if cv2.imread(str(path)) is None:
              raise ValueError(f'Invalid RGB file {path}')
          else:
            with np.load(path, allow_pickle=False) as data:
              points = data['points']
              if points.ndim != 2 or points.shape[1] < 3 or not np.isfinite(points).all():
                raise ValueError(f'Invalid LiDAR file {path}')
              if int(data['sensor_frame']) != row['sensor_frames']['lidar']:
                raise ValueError(f'LiDAR frame mismatch {path}')
      count += 1
      last_frame, last_step = frame, step
  with (directory / 'events.jsonl').open() as handle:
    events = [json.loads(line) for line in handle]
  if count == 0 or count != summary['frame_count'] or predictions != summary['prediction_count']:
    raise ValueError(f'Frame/prediction count mismatch in {directory}')
  if samples != summary['sensor_sample_count'] or len(events) != summary['event_record_count']:
    raise ValueError(f'Sensor/event count mismatch in {directory}')
  if require_prediction and predictions == 0:
    raise ValueError(f'No model predictions saved in {directory}')
  return {'directory': str(directory), 'frames': count, 'predictions': predictions,
          'sensor_samples': samples, 'events': len(events)}
