"""Speed-only optimizer integrity and label-free rollout prefix regressions."""
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
import torch
from torch import nn
from scripts.pdm_lite.bounded_speed_recovery import speed_rows, integrity, recovery_loss


class SpeedRowsTests(unittest.TestCase):
    def test_optimizer_updates_only_speed_entries_in_both_layers(self):
        head = nn.Module()
        head.config = SimpleNamespace(diffusion_speed=True)
        head._num_poses = 10
        head.encoder = nn.Linear(8, 8)
        head.diff_decoder = nn.Module()
        head.diff_decoder.layers = nn.ModuleList()
        for _ in range(2):
            layer = nn.Module(); layer.task_decoder = nn.Module()
            layer.task_decoder.plan_reg_branch = nn.Sequential(
                nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 33))
            head.diff_decoder.layers.append(layer)
        params, masks, effective = speed_rows(head)
        initial = {k: v.detach().clone() for k, v in head.state_dict().items()}
        optimizer = torch.optim.AdamW(params, lr=.01, weight_decay=0.)
        output = sum(layer.task_decoder.plan_reg_branch(torch.randn(4, 8)).sum()
                     for layer in head.diff_decoder.layers)
        output.backward(); optimizer.step()
        _, changed = integrity(head, initial, masks)
        self.assertEqual(effective, 18)
        self.assertEqual(len(changed), 4)
        self.assertFalse(head.encoder.weight.requires_grad)
        for layer in head.diff_decoder.layers:
            self.assertFalse(layer.task_decoder.plan_reg_branch[0].weight.requires_grad)

    def test_speed_labels_only_affect_loss_and_not_generated_prefix(self):
        class FakeHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.plan_anchor = nn.Parameter(torch.ones(3, 11, 2), requires_grad=False)
                self.offset = nn.Parameter(torch.tensor(.2))
                self.config = SimpleNamespace(flow_inference_steps=20)
                self.states = []
            def _deterministic_cr_decoder(self):
                return nullcontext()
            def _classify_flow_anchors(self, anchors, *_):
                return [torch.zeros(len(anchors), 3)]
            def gather_mode(self, anchors, mode):
                return anchors[torch.arange(len(anchors)), mode]
            def _flow_rollout_query(self, anchors, indices, *_):
                self.states.append(anchors.clone())
                return anchors.clone(), torch.full((len(anchors),), .5)
            def predict_flow_endpoint(self, state, *_):
                return state+self.offset
        head = FakeHead(); query = torch.randn(2, 11, 8); bev = torch.randn(2, 8, 2, 2)
        torch.manual_seed(137)
        first = recovery_loss(head, query, bev, torch.zeros(2, 11, 2), scenes=2)
        first.backward()
        torch.manual_seed(137)
        second = recovery_loss(head, query, bev, torch.full((2, 11, 2), 99.), scenes=2)
        torch.testing.assert_close(head.states[0], head.states[1], atol=0, rtol=0)
        self.assertNotEqual(first.item(), second.item())
        self.assertTrue(torch.isfinite(head.offset.grad))
        self.assertGreater(head.offset.grad.abs().item(), 0)


if __name__ == '__main__':
    unittest.main()
