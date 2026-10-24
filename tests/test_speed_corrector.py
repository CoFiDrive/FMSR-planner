import unittest
import torch
from fmsr.tfv6.diffusion_modules.speed_corrector import SpeedCorrector


class SpeedCorrectorTests(unittest.TestCase):
    def test_zero_initialization_preserves_prediction_exactly(self):
        model = SpeedCorrector(embed_dims=8)
        query = torch.randn(3, 11, 8)
        prediction = torch.randn(3, 11, 2)
        prediction[:, -1, 0] = torch.tensor([0., 4., 24.])
        torch.testing.assert_close(model(query, prediction), prediction, atol=0, rtol=0)

    def test_training_changes_speed_only_without_source_gradients(self):
        model = SpeedCorrector(embed_dims=8)
        query = torch.randn(3, 11, 8, requires_grad=True)
        prediction = torch.randn(3, 11, 2, requires_grad=True)
        with torch.no_grad():
            prediction[:, -1, 0] = 4.
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=0.)
        loss = (model.raw_speed(query, prediction)-8.).square().mean()
        loss.backward(); optimizer.step()
        result = model(query, prediction)
        torch.testing.assert_close(result[:, :10], prediction[:, :10], atol=0, rtol=0)
        torch.testing.assert_close(result[:, -1, 1], prediction[:, -1, 1], atol=0, rtol=0)
        self.assertIsNone(query.grad)
        self.assertIsNone(prediction.grad)
        self.assertGreater((result[:, -1, 0]-prediction[:, -1, 0]).abs().sum().item(), 0)

    def test_inference_speed_bounds(self):
        model = SpeedCorrector(embed_dims=8)
        query = torch.randn(2, 11, 8); prediction = torch.randn(2, 11, 2)
        prediction[:, -1, 0] = torch.tensor([-10., 40.])
        torch.testing.assert_close(model(query, prediction)[:, -1, 0], torch.tensor([0., 30.]))


if __name__ == '__main__':
    unittest.main()
