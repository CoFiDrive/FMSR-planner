"""Route-level physical telemetry and spatially aligned replanning diagnostics."""
import argparse
import json
from pathlib import Path
import numpy as np


def statistics(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return {'count': len(values), 'mean': float(values.mean()),
            'rms': float(np.sqrt(np.mean(values ** 2))),
            'p95_abs': float(np.percentile(abs(values), 95)), 'max_abs': float(abs(values).max())}


def temporal_statistics(values, timestamp):
    """Exploratory empirical ACF/PSD with an explicit detrending/resampling protocol."""
    from scipy.signal import detrend, welch
    values = np.asarray(values, dtype=float)
    timestamp = np.asarray(timestamp, dtype=float)
    if len(values) < 8 or timestamp[-1] <= timestamp[0]:
        return None
    dt = float(np.median(np.diff(timestamp)))
    grid = np.arange(timestamp[0], timestamp[-1]+dt*.1, dt)
    signal = detrend(np.interp(grid, timestamp, values), type='linear')
    energy = float(signal @ signal)
    limit = min(len(signal)//2, max(1, round(5/dt)))
    if energy < 1e-12:
        acf = None
    else:
        acf = [float(signal[:len(signal)-lag] @ signal[lag:] / energy) for lag in range(limit+1)]
    frequency, psd = welch(signal, fs=1/dt, nperseg=min(256,len(signal)))
    return {'protocol':'linear interpolation to median dt; whole-route linear detrend; '
                       'biased normalized empirical ACF <=5s; Welch PSD; exploratory, not GP kernel',
            'sample_interval_s':dt, 'acf_lags_s':(np.arange(limit+1)*dt).tolist(),
            'acf':acf, 'frequency_hz':frequency.tolist(), 'psd':psd.tolist()}


def world_path(points, state):
    points = np.asarray(points, dtype=float)
    yaw = np.deg2rad(state['rotation'][2])
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    origin = np.asarray(state['location'][:2])
    return np.concatenate([origin[None], points @ rotation.T + origin], axis=0)


def spatial_deviation(previous, current):
    # Advance along the previous path to the projection of the current ego.
    vector = np.diff(previous, axis=0)
    lengths = np.linalg.norm(vector, axis=1)
    if np.any(lengths < 1e-6):
        previous = previous[np.r_[True, lengths > 1e-6]]
        vector = np.diff(previous, axis=0)
        lengths = np.linalg.norm(vector, axis=1)
    if len(lengths) < 1:
        return None
    fraction = np.clip(np.sum((current[0] - previous[:-1]) * vector, axis=1) / lengths**2, 0, 1)
    projections = previous[:-1] + vector * fraction[:, None]
    segment = np.argmin(np.linalg.norm(projections - current[0], axis=1))
    old_s = np.r_[0, np.cumsum(lengths)]
    progress = old_s[segment] + lengths[segment] * fraction[segment]
    new_s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(current, axis=0), axis=1))]
    current = current[np.r_[True, np.diff(new_s) > 1e-6]]
    new_s = new_s[np.r_[True, np.diff(new_s) > 1e-6]]
    end = min(old_s[-1] - progress, new_s[-1])
    if end <= 2.5:
        return None
    grid = np.linspace(2.5, end, 20)
    old = np.stack([np.interp(progress + grid, old_s, previous[:, axis]) for axis in range(2)], axis=-1)
    new = np.stack([np.interp(grid, new_s, current[:, axis]) for axis in range(2)], axis=-1)
    distance = np.linalg.norm(new - old, axis=-1)
    return {'mean_m': float(distance.mean()), 'max_m': float(distance.max()),
            'overlap_length_m': float(end - 2.5)}


def analyze(directory):
    directory = Path(directory)
    states = [json.loads(line) for line in (directory / 'vehicle_state.jsonl').open()]
    plans = [json.loads(line) for line in (directory / 'planning.jsonl').open()]
    indexed = {state['step']: state for state in states}
    timestamp = np.asarray([s['timestamp_s'] for s in states])
    if len(timestamp) < 3 or np.any(np.diff(timestamp) <= 0):
        raise ValueError('Need >=3 increasing state timestamps')
    acceleration = np.asarray([s['acceleration'] for s in states])
    forward = np.asarray([s['forward_vector'] for s in states])
    right = np.asarray([s['right_vector'] for s in states])
    lon = np.sum(acceleration[:, :2] * forward[:, :2], axis=-1)
    lat = np.sum(acceleration[:, :2] * right[:, :2], axis=-1)
    yaw_rate = np.deg2rad([s['angular_velocity'][2] for s in states])
    series = []
    previous = None
    for row in plans:
        policy = row['policy']
        if policy.get('phase') != 'inference':
            continue
        state = indexed[row['step']]
        path = policy.get('predicted_trajectory_ego_m')
        generation = policy.get('generation', [])
        modes = [g.get('selected_mode') for g in generation]
        entry = {'step': row['step'], 'timestamp_s': row['timestamp_s'],
                 'mode_ids': modes, 'spatial_plan_deviation': None, 'mode_switched': None,
                 'overrides': policy.get('overrides', []),
                 'control_target_speed_mps': policy.get('control_target_speed_mps'),
                 'agent_policy_wall_ms': policy.get('agent_policy_wall_ms'),
                 'planner_gpu_ms': [g.get('planner_gpu_ms') for g in generation],
                 'nfe': [g.get('nfe') for g in generation]}
        if path is not None:
            current_path = world_path(path, state)
            if previous is not None:
                entry['spatial_plan_deviation'] = spatial_deviation(previous[0], current_path)
                if modes and previous[1] and all(m is not None for m in modes + previous[1]):
                    entry['mode_switched'] = modes != previous[1]
            previous = current_path, modes
        series.append(entry)
    switch = [s['mode_switched'] for s in series if s['mode_switched'] is not None]
    return {'save_name': directory.name, 'physical_diagnostic_protocol':
            'raw CARLA acceleration; angular rate converted deg/s to rad/s; '
            'numpy.gradient using actual timestamps, no smoothing; distinct from official Comfort',
            'trajectory_alignment': 'world pose and projected spatial arclength; not time-aligned Eq29 E_t',
            'time_interval_s': statistics(np.diff(timestamp)),
            'longitudinal_acceleration_mps2': statistics(lon),
            'lateral_acceleration_mps2': statistics(lat),
            'longitudinal_jerk_mps3': statistics(np.gradient(lon, timestamp)),
            'lateral_jerk_mps3': statistics(np.gradient(lat, timestamp)),
            'yaw_rate_radps': statistics(yaw_rate),
            'yaw_acceleration_radps2': statistics(np.gradient(yaw_rate, timestamp)),
            'mode_switch_fraction': sum(switch) / len(switch) if switch else None,
            'spatial_plan_deviation_m': statistics([s['spatial_plan_deviation']['mean_m'] for s in series
                                                   if s['spatial_plan_deviation']]),
            'agent_policy_wall_ms': statistics([s['agent_policy_wall_ms'] for s in series
                                               if s['agent_policy_wall_ms'] is not None]),
            'planner_gpu_ms': statistics([v for s in series for v in s['planner_gpu_ms'] if v is not None]),
            'longitudinal_acceleration_temporal': temporal_statistics(lon,timestamp),
            'lateral_acceleration_temporal': temporal_statistics(lat,timestamp),
            'series': series}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metric-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = []
    for directory in sorted(args.metric_dir.iterdir()):
        if directory.is_dir() and (directory / 'vehicle_state.jsonl').exists():
            try:
                output.append(analyze(directory))
            except ValueError as error:
                output.append({'save_name': directory.name, 'error': str(error)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'routes': output}, ensure_ascii=False, indent=2) + '\n')
    print(f'Wrote {len(output)} route analyses to {args.output}')


if __name__ == '__main__':
    main()
