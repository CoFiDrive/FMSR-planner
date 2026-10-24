"""Exact-input capture preserves outputs and isolates saved tensors."""
import sys
from pathlib import Path
import tempfile
import gzip
import json
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'3rd_party/carla_garage/team_code'),str(ROOT),str(ROOT/'3rd_party/CARLA_0915/PythonAPI/carla')]
sys.path[:0]=[str(ROOT/'3rd_party/Bench2Drive/leaderboard'),str(ROOT/'3rd_party/Bench2Drive/scenario_runner')]
from native_failure_capture_agent import NativeFailureCaptureAgent
from sensor_agent import SensorAgent
from scripts.pdm_lite.recovery_data import valid_label


class CaptureTests(unittest.TestCase):
    def test_unreviewed_reference_labels_are_excluded_from_training_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name,extension in [('lidar','laz'),('rgb','jpg')]:
                (root/name).mkdir();(root/name/f'0000.{extension}').touch()
            (root/'measurements').mkdir()
            label={'label_source':'learned_teacher','label_valid':True,'training_eligible':False,
                   'target_speed':5.,'route':[[float(i),0.] for i in range(10)]}
            with gzip.open(root/'measurements/0000.json.gz','wt') as f:json.dump(label,f)
            self.assertFalse(valid_label(root,0))
            label.pop('training_eligible')
            with gzip.open(root/'measurements/0000.json.gz','wt') as f:json.dump(label,f)
            self.assertTrue(valid_label(root,0))

    def test_capture_preserves_predictions_and_copies_only_sampled_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'model_test.pth').touch()
            result=object();calls=[]
            def forward(**kwargs):calls.append(kwargs);return result
            net=SimpleNamespace(forward=forward)
            agent=NativeFailureCaptureAgent.__new__(NativeFailureCaptureAgent)
            agent.config_path=str(root);agent.config=SimpleNamespace(use_wp_gru=False,detect_boxes=False,data_save_freq=5)
            agent.nets=[net];agent.step=1
            with patch.object(SensorAgent,'setup'),patch.dict('os.environ',{'SHADOW_COLLECTION_ROOT':str(root/'saved'),'SHADOW_TEACHER_CHECKPOINT':'teacher.pth'}):
                agent.setup(str(root)+'+RouteScenario_1_rep0_Test')
            x=torch.ones(1,3)
            self.assertIs(net.forward(rgb=x),result);self.assertIsNone(agent.capture_inputs)
            agent.step=5
            self.assertIs(net.forward(rgb=x),result)
            saved=agent.capture_inputs['rgb'];x.add_(2)
            torch.testing.assert_close(saved,torch.ones(1,3))
            self.assertIs(calls[-1]['rgb'],x)
            agent.capture_writer.close()


if __name__=='__main__':unittest.main()
