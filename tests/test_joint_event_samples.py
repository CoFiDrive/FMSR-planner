"""Checks for cold-start exclusion and bounded repeated-stop event selection."""
import unittest

from scripts.pdm_lite.joint_event_samples import select_window_rows, trigger_windows


class EventSelectionTests(unittest.TestCase):
    def test_persistent_disagreement_does_not_select_entire_stationary_route(self):
        rows = [{'frame': i, 'time_s': i * .25, 'eligible': i >= 10,
                 'reasons': ['student_stop_teacher_go']} for i in range(401)]
        selected = select_window_rows(rows, trigger_windows(rows))
        self.assertIn(10, selected)
        self.assertIn(400, selected)
        self.assertNotIn(0, selected)
        self.assertNotIn(200, selected)
        self.assertLessEqual(len(selected), 20)

    def test_opposite_decisions_keep_distinct_windows_and_exclude_invalid_labels(self):
        rows = [{'frame': i, 'time_s': i * .25, 'eligible': True,
                 'reasons': []} for i in range(61)]
        rows[12]['reasons'] = ['student_go_teacher_stop']
        rows[48]['reasons'] = ['student_stop_teacher_go']
        rows[11]['eligible'] = False
        selected = select_window_rows(rows, trigger_windows(rows))
        self.assertEqual(selected[12]['window_reasons'], ['student_go_teacher_stop'])
        self.assertEqual(selected[48]['window_reasons'], ['student_stop_teacher_go'])
        self.assertNotIn(11, selected)
        self.assertNotIn(30, selected)


if __name__ == '__main__':
    unittest.main()
