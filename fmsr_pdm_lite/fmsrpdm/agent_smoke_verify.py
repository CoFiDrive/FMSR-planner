"""Verify agent setup, real CUDA prediction and PID without connecting to CARLA."""
import argparse
import json
import math
import os
from pathlib import Path

import torch
from torch.utils.data import default_collate

from .agent import FMSRPDMAgent, get_entry_point
from .configuration import RunConfig
from .dataset import create_dataset
from .train import tensor_inputs
from .bootstrap import GARAGE_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    os.environ.update(IS_BENCH2DRIVE='1', DIRECT='1', COMPILE='0',
                      FMSR_DIAGNOSTICS='0', DEBUG_CHALLENGE='0')
    if not torch.cuda.is_available():
        raise RuntimeError('This integration check needs CUDA')
    torch.set_num_threads(2)
    # Exercise setup directly: the leaderboard constructor otherwise asks for
    # the live hero actor. This check intentionally has no simulator connection.
    agent = FMSRPDMAgent.__new__(FMSRPDMAgent)
    agent.setup(str(args.agent_dir.resolve()))
    try:
        assert agent.model_count == 1 and len(agent.nets) == 1
        run = RunConfig(**json.loads((args.run_dir/'run_config.json').read_text()))
        dataset = create_dataset(run, agent.config, validation=True)
        batch = default_collate([dataset[0]])
        inputs = tensor_inputs(batch, agent.device)
        for key in ('return_loss', 'target_route', 'target_speed'):
            inputs.pop(key)
        with torch.inference_mode():
            prediction = agent.nets[0](**inputs)
        speed, route = prediction[1], prediction[2]
        assert speed.shape == (1, 1) and route.shape == (1, 10, 2)
        assert torch.isfinite(route).all() and torch.isfinite(speed).all()
        controls = agent.nets[0].control_pid_direct(
            route[0].cpu().numpy(), float(speed[0, 0]), inputs['ego_vel'].reshape(-1))
        assert all(math.isfinite(float(value)) for value in controls)
        steer, throttle, brake = (float(value) for value in controls)
        assert -1 <= steer <= 1 and 0 <= throttle <= 1 and 0 <= brake <= 1
        # A stopped prediction only exercises get_throttle's early return. The
        # fitted controller must also work at cruising speed and during braking.
        controller_cases = {}
        for name, target, current in [('accelerating', 8., 0.), ('cruising', 5., 5.),
                                      ('overspeed', 5., 5.3), ('stopping', 0., 2.)]:
            values = agent.nets[0].control_pid_direct(
                route[0].cpu().numpy(), target, torch.tensor([current], device=agent.device))
            assert all(math.isfinite(float(value)) for value in values)
            case_steer, case_throttle, case_brake = (float(value) for value in values)
            assert -1 <= case_steer <= 1 and 0 <= case_throttle <= 1 and 0 <= case_brake <= 1
            controller_cases[name] = {'steer': case_steer, 'throttle': case_throttle, 'brake': case_brake}
        import sensor_agent
        assert Path(sensor_agent.__file__).resolve().is_relative_to(GARAGE_ROOT)
        report = {
            'status': 'passed', 'entry_point': get_entry_point(),
            'strict_agent_model_loading': True, 'models_loaded': agent.model_count,
            'sensor_agent_source': sensor_agent.__file__,
            'sensor_ids': [sensor['id'] for sensor in agent.sensors()],
            'device': str(speed.device), 'inference_steps': agent.config.step_num,
            'predicted_route_shape': list(route.shape),
            'target_speed_mps': float(speed[0, 0]),
            'controls': {'steer': steer, 'throttle': throttle, 'brake': brake},
            'controller_cases': controller_cases,
            'longitudinal_params_type': type(agent.config.longitudinal_params).__name__,
            'connected_to_simulator': False,
        }
        (args.run_dir/'agent_smoke_verification.json').write_text(
            json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
    finally:
        agent.destroy()


if __name__ == '__main__':
    main()
