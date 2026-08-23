from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from train.grpo import RewardFn, validate_generation_batch


class RewardResumeTest(unittest.TestCase):
    def test_resume_truncates_noncheckpointed_curve_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "curve.jsonl"
            path.write_text('{"step": 0}\n{"step": 1}\n{"step": 2}\n')
            reward = RewardFn("partial", 8, str(path), resume_step=2)
            self.assertEqual(reward.step, 2)
            self.assertEqual(len(path.read_text().splitlines()), 2)

    def test_resume_requires_a_matching_curve(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing.jsonl"
            with self.assertRaises(FileNotFoundError):
                RewardFn("binary", 8, str(path), resume_step=1)


class GRPOBatchBudgetTest(unittest.TestCase):
    def test_memory_safe_microbatch_preserves_rollout_budget(self):
        self.assertEqual(validate_generation_batch(8, 4, 8, 32), 32)
        self.assertEqual(validate_generation_batch(4, 8, 8, 32), 32)

    def test_changed_rollout_budget_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "rollout budget changed"):
            validate_generation_batch(4, 4, 8, 32)

    def test_generation_groups_must_be_integral(self):
        with self.assertRaisesRegex(ValueError, "not divisible"):
            validate_generation_batch(3, 10, 8, 30)


if __name__ == "__main__":
    unittest.main()
