"""Protect braking/restart transitions while reducing repeated stopped states."""
import unittest

from scripts.pdm_lite.stage_joint_posttraining import stationary_frame_weights


class StationaryWeightsTests(unittest.TestCase):
    def test_long_stop_keeps_entry_and_exit_windows(self):
        events = {i: {'speed': 0.} for i in range(40)}
        weights = stationary_frame_weights(events, .25)
        self.assertTrue(all(weights[i] == 1 for i in range(20)))
        self.assertTrue(all(weights[i] == .1 for i in range(20, 36)))
        self.assertTrue(all(weights[i] == 1 for i in range(36, 40)))

    def test_moving_and_missing_frames_reset_stationary_duration(self):
        events = {i: {'speed': 0.} for i in range(60)}
        events[20]['speed'] = -1.
        del events[40]
        weights = stationary_frame_weights(events, .25)
        self.assertTrue(all(weight == 1 for weight in weights.values()))


if __name__ == '__main__':
    unittest.main()
