from __future__ import annotations

import unittest

from analysis.win_matrix import win_matrix


class WinMatrixTest(unittest.TestCase):
    def test_all_four_paired_cells_and_orientation(self):
        teacher = {"a": [True], "b": [True], "c": [False], "d": [False]}
        student = {"a": [True], "b": [False], "c": [True], "d": [False]}
        result = win_matrix(teacher, student)
        self.assertEqual(result["matrix"], [[1, 1], [1, 1]])
        self.assertEqual(result["task_ids"]["both_solve"], ["a"])
        self.assertEqual(result["task_ids"]["teacher_only"], ["b"])
        self.assertEqual(result["task_ids"]["student_only"], ["c"])
        self.assertEqual(result["task_ids"]["neither_solves"], ["d"])

    def test_any_sample_changes_reachability_criterion(self):
        teacher = {"a": [False, True], "b": [False, False]}
        student = {"a": [False, False], "b": [False, True]}
        result = win_matrix(teacher, student, any_sample=True)
        self.assertEqual(result["matrix"], [[0, 1], [1, 0]])

    def test_task_set_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "task sets differ"):
            win_matrix({"a": [True]}, {"b": [True]})


if __name__ == "__main__":
    unittest.main()
