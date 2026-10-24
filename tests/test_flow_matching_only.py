"""FM-only copy matches current CR-disabled math and retains trainable layers."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile
import sys
import numpy as np
import torch
from fmsr.tfv6.diffusion_modules.model_diffusion_head_tie_cr_fd import TrajectoryHead as Original
from fmsr.tfv6.diffusion_modules.model_flow_matching_only import TrajectoryHead as FlowOnly

ROOT=Path(__file__).resolve().parents[1]


class FlowOnlyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(42)
        self.cfg=SimpleNamespace(trajectory_generator='residual_flow',diffusion_speed=False,
            step_num=20,flow_train_t_max=.95,flow_min_remain=.05,flow_inference_steps=20,
            flow_contract_enable=False,flow_contract_method='endpoint_jacobian',flow_contract_warmup_steps=0,
            tf_d_model=256,tf_d_ffn=128,tf_num_head=8,tf_dropout=0.,lidar_max_x=32.,lidar_max_y=32.)
        path=str(ROOT/'anchor_utils/anchor_data/fmsr_cp_kmeans_60_10.npy')
        self.old=Original(10,128,256,path,self.cfg)
        self.new=FlowOnly(10,128,256,path,self.cfg)
        self.new.load_state_dict(self.old.state_dict(),strict=True)
        self.query=torch.randn(1,10,256)*.1;self.bev=torch.randn(1,1512,2,2)*.1

    def test_checkpoint_keys_and_complete_generation_are_identical(self):
        self.assertEqual(set(self.old.state_dict()),set(self.new.state_dict()))
        self.old.eval();self.new.eval()
        with torch.no_grad():
            expected=self.old(self.query,self.bev,(2,2));actual=self.new(self.query,self.bev,(2,2))
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        self.assertFalse(hasattr(self.new,'contractive_recovery_loss'))
        self.assertFalse(hasattr(self.new,'_spectral_contract_at_state'))
        self.assertFalse(hasattr(self.new,'diffusion_scheduler'))

    def test_fm_loss_matches_original_and_backward_updates_both_layers(self):
        self.old.train();self.new.train();gt=self.old.plan_anchor[3:4]+.1
        torch.manual_seed(7);expected=self.old.forward_train_flow(self.query,self.bev,(2,2),gt_checkpoints=gt)
        torch.manual_seed(7);actual=self.new.forward_train_flow(self.query,self.bev,(2,2),gt_checkpoints=gt)
        torch.testing.assert_close(actual['flow_loss'],expected['flow_loss'],rtol=0,atol=0)
        self.assertFalse(any('contract' in key or 'recovery' in key for key in actual))
        actual['flow_loss'].backward()
        for layer in self.new.diff_decoder.layers:
            grads=[p.grad for name,p in layer.named_parameters() if 'plan_cls_branch' not in name]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(float(g.abs().sum()) for g in grads),0.)
        self.assertIsNone(self.new.plan_anchor.grad)

    def test_enabling_cr_is_rejected(self):
        cfg=copy.copy(self.cfg);cfg.flow_contract_enable=True
        with self.assertRaisesRegex(ValueError,'no CR'):
            FlowOnly(10,128,256,str(ROOT/'anchor_utils/anchor_data/fmsr_cp_kmeans_60_10.npy'),cfg)

    def test_native_training_loss_and_optimizer_with_appended_speed(self):
        sys.path[:0]=[str(ROOT/'3rd_party/carla_garage/team_code'),str(ROOT/'3rd_party/CARLA_0915/PythonAPI/carla')]
        from model import LidarCenterNet
        from fmsr.tfv6.diffusion_modules.multimodal_loss_ddbm import LossComputer
        cfg=copy.copy(self.cfg);cfg.diffusion_speed=True;cfg.fmsr_num_poses=10
        anchors=np.concatenate((self.old.plan_anchor.numpy(),np.zeros((60,1,2),dtype=np.float32)),axis=1)
        anchors[:,-1,0]=5.
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'anchors.npy';np.save(path,anchors)
            head=FlowOnly(10,128,256,str(path),cfg).train()
            target=head.plan_anchor[3:4].clone();target[:,:10]+=.1
            output=head(self.query,self.bev,(2,2),gt_checkpoints=target)
            owner=SimpleNamespace(config=cfg,fmsr_head=head,fmsr_loss_computer=LossComputer(cfg))
            losses,metrics=LidarCenterNet.compute_fmsr_loss(owner,output,target)
            torch.testing.assert_close(losses['loss_fmsr_reg'],output['flow_loss'])
            self.assertFalse(any('contract' in key or 'recovery' in key for key in metrics))
            optimizer=torch.optim.AdamW([p for p in head.parameters() if p.requires_grad],lr=1e-4)
            before=head.time_mlp[1].weight.detach().clone()
            sum(losses.values()).backward();optimizer.step()
            self.assertTrue(torch.isfinite(sum(losses.values())))
            self.assertFalse(torch.equal(before,head.time_mlp[1].weight))


if __name__=='__main__':unittest.main()
