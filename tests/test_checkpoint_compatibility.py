"""Checkpoint loading must bypass downloads without accepting random encoders."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'3rd_party/carla_garage/team_code'))
from checkpoint_compatibility import configure_local_checkpoint_initialization,validate_image_encoder_checkpoint


class CheckpointCompatibilityTests(unittest.TestCase):
    def test_bridge_local_checkpoint_disables_online_pretraining(self):
        cfg=SimpleNamespace(use_fmsr=True,image_encoder_pretrained=True)
        configure_local_checkpoint_initialization(cfg,'local_tfpp.pth')
        self.assertFalse(cfg.image_encoder_pretrained)
        for use,load in ((True,None),(False,'local.pth')):
            c=SimpleNamespace(use_fmsr=use,image_encoder_pretrained=True)
            configure_local_checkpoint_initialization(c,load)
            self.assertTrue(c.image_encoder_pretrained)

    def test_full_image_weights_required_and_old_bn_counters_optional(self):
        model={'backbone.image_encoder.conv.weight':torch.zeros(2,3),
               'backbone.image_encoder.bn.running_mean':torch.zeros(2),
               'backbone.image_encoder.bn.num_batches_tracked':torch.zeros((),dtype=torch.long)}
        state={k:v.clone() for k,v in model.items() if not k.endswith('num_batches_tracked')}
        self.assertEqual(validate_image_encoder_checkpoint(model,state),2)
        missing=dict(state);missing.pop('backbone.image_encoder.conv.weight')
        with self.assertRaisesRegex(RuntimeError,'missing='):validate_image_encoder_checkpoint(model,missing)
        wrong=dict(state);wrong['backbone.image_encoder.conv.weight']=torch.zeros(3,3)
        with self.assertRaisesRegex(RuntimeError,'shape_mismatch='):validate_image_encoder_checkpoint(model,wrong)


if __name__=='__main__':unittest.main()
