from pathlib import Path
from types import SimpleNamespace as NS
import hashlib
import json

import jsonpickle

import numpy as np
import pytest
import torch

from fmsrpdm.dataset import validation_subset
from fmsrpdm.model import FMSRModel
from fmsrpdm.speed import continuous_speed_limit
from fmsr_interface import continuous_speed_for_control
from fmsrpdm import bootstrap, stage_evaluation


def test_continuous_highway_speed_survives_model_and_agent_bounds():
    prediction = torch.zeros(2, 11, 2)
    prediction[:, -1, 0] = torch.tensor([24., 35.])
    config = NS(target_speeds=[0., 5., 10., 20.], detect_boxes=False)
    fake_model = NS(config=config,
                    context=lambda *args: (torch.zeros(2, 11, 256), torch.zeros(2, 4, 2, 2), None),
                    planner=NS(sample=lambda *args, **kwargs: (prediction, torch.zeros(2))))
    output = FMSRModel.forward(fake_model, None, None, None, None, None)
    torch.testing.assert_close(output[1], torch.tensor([[24.], [30.]]))
    assert continuous_speed_for_control([output[1][0]], continuous_speed_limit(config)) == 24.


def test_explicit_continuous_bound_and_invalid_configuration():
    assert continuous_speed_limit(NS(fmsr_max_target_speed_mps=25.)) == 25.
    for value in (0., -1., float('inf'), float('nan')):
        with pytest.raises(ValueError):
            continuous_speed_limit(NS(fmsr_max_target_speed_mps=value))


class OrderedScenarios:
    def __init__(self):
        self.measurements = np.array([[str(Path('/data') / name / 'route' / 'measurements').encode()]
                                     for name in ('A', 'B', 'C') for _ in range(100)])

    def __len__(self):
        return len(self.measurements)


def test_bounded_validation_covers_later_scenarios_deterministically():
    data = OrderedScenarios()
    chosen = validation_subset(data, 30)
    assert len(chosen) == len(set(chosen.indices)) == 30
    assert {Path(data.measurements[i][0].decode()).parent.parent.name for i in chosen.indices} == {'A', 'B', 'C'}
    assert chosen.indices == validation_subset(data, 30).indices
    assert validation_subset(data, 0) is data
    assert validation_subset(data, len(data)) is data
    with pytest.raises(ValueError, match='scenario count'):
        validation_subset(data, 2)


def test_speed_fix_cannot_resume_old_evaluation_results(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'model_0009.pth'
    checkpoint.write_bytes(b'test checkpoint')
    config = tmp_path / 'config.json'
    config.write_text(jsonpickle.encode(NS(fmsr_trajectory_head='paper_vp_ddbm', step_num=20)))
    routes = tmp_path / 'routes.xml'
    routes.write_text('<routes><route id="1"/></routes>')
    output = tmp_path / 'evaluation'
    output.mkdir()
    old_settings = {'checkpoint': str(checkpoint),
                    'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    'config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
                    'routes': str(routes), 'routes_sha256': hashlib.sha256(routes.read_bytes()).hexdigest(),
                    'route_count': 1, 'agent': str(bootstrap.PACKAGE_ROOT / 'fmsrpdm/agent.py'),
                    'speed_interface': 'continuous_mps', 'inference_steps': 20,
                    'traffic_manager_seed': 0, 'perception_source': 'isolated_garage_snapshot'}
    source = output / 'source.json'
    source.write_text(json.dumps(old_settings))
    before = source.read_bytes()
    monkeypatch.setattr('sys.argv', ['stage_evaluation', '--checkpoint', str(checkpoint),
                                    '--routes', str(routes), '--output', str(output)])
    with pytest.raises(ValueError, match='different model/config/routes'):
        stage_evaluation.main()
    assert source.read_bytes() == before
    assert not (output / 'agent').exists()


def test_experimental_clean_prior_evaluation_requires_explicit_flag_and_versions_metadata(tmp_path,monkeypatch):
    checkpoint=tmp_path/'model_0009.pth';checkpoint.write_bytes(b'test')
    (tmp_path/'config.json').write_text(jsonpickle.encode(NS(
        fmsr_trajectory_head='experimental_tfpp_clean_prior_v1',
        prediction_base='tfpp_clean_prior_v1',step_num=20)))
    routes=tmp_path/'routes.xml';routes.write_text('<routes><route id="1"/></routes>')
    output=tmp_path/'evaluation'
    arguments=['stage_evaluation','--checkpoint',str(checkpoint),'--routes',str(routes),'--output',str(output)]
    monkeypatch.setattr('sys.argv',arguments)
    with pytest.raises(ValueError,match='Explicit'):
        stage_evaluation.main()
    monkeypatch.setattr('sys.argv',arguments+['--allow-experimental-clean-prior'])
    stage_evaluation.main()
    source=json.loads((output/'source.json').read_text())
    assert source['experimental_architecture']
    assert source['prediction_base']=='tfpp_clean_prior_v1'
    assert source['trajectory_head']=='experimental_tfpp_clean_prior_v1'
    assert source['carla_client_timeout_seconds']==120.
    assert list((output/'agent').glob('model*.pth'))==[output/'agent/model_0009.pth']
