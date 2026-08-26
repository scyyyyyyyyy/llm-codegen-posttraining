from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from train.opd import (
    completion_prediction_positions,
    latest_checkpoint,
    prepare_curve,
    reverse_kl_loss,
    rollout_schedule,
    sampled_reverse_kl,
    selective_log_softmax,
)


class ReverseKLTest(unittest.TestCase):
    def test_on_policy_value_and_gradient_use_completion_tokens_only(self):
        student = torch.tensor([[-1.0, -2.0, -3.0]], requires_grad=True)
        teacher = torch.tensor([[-2.0, -1.0, -100.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0]])

        # The two selected sampled KL values are +1 and -1. Their mean/loss is
        # zero, but the policy-gradient directions remain non-zero.
        estimate = sampled_reverse_kl(student, teacher, mask)
        loss = reverse_kl_loss(student, teacher, mask)
        self.assertAlmostEqual(float(estimate), 0.0)
        self.assertAlmostEqual(float(loss), 0.0)

        loss.backward()
        torch.testing.assert_close(
            student.grad, torch.tensor([[0.5, -0.5, 0.0]])
        )

    def test_explicit_behavior_policy_uses_importance_ratio(self):
        current = torch.tensor([[-1.5, -0.5]], requires_grad=True)
        sampled = torch.tensor([[-2.0, -1.0]])
        teacher = torch.tensor([[-3.0, -0.5]])
        mask = torch.ones_like(current)

        loss = reverse_kl_loss(current, teacher, mask, sampled)
        expected = torch.exp(torch.tensor(0.5)) * torch.tensor(0.25)
        torch.testing.assert_close(loss.detach(), expected)

    def test_empty_mask_is_rejected(self):
        values = torch.zeros((1, 2), requires_grad=True)
        with self.assertRaisesRegex(ValueError, "zero tokens"):
            reverse_kl_loss(values, values.detach(), torch.zeros_like(values))


class TokenAlignmentTest(unittest.TestCase):
    def test_selective_log_softmax_matches_full_reference(self):
        logits = torch.tensor(
            [[[1.0, 2.0, 3.0], [4.0, -1.0, 0.0]]], requires_grad=True
        )
        ids = torch.tensor([[2, 0]])
        actual = selective_log_softmax(logits, ids)
        expected = logits.log_softmax(dim=-1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(actual, expected)

    def test_completion_positions_are_shifted_once(self):
        positions = completion_prediction_positions(4, 7)
        torch.testing.assert_close(positions, torch.tensor([3, 4, 5]))
        with self.assertRaisesRegex(ValueError, "no completion"):
            completion_prediction_positions(4, 4)


class ScheduleAndResumeTest(unittest.TestCase):
    def test_schedule_is_deterministic_balanced_and_seeded(self):
        first = rollout_schedule(7, 3, seed=2)
        self.assertEqual(first, rollout_schedule(7, 3, seed=2))
        self.assertNotEqual(first, rollout_schedule(7, 3, seed=3))
        self.assertEqual(len(first), 21)
        self.assertEqual([first.count(i) for i in range(7)], [3] * 7)

    def test_curve_is_truncated_to_checkpoint_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "curve.jsonl"
            path.write_text("a\nb\nc\n")
            prepare_curve(path, 2)
            self.assertEqual(path.read_text(), "a\nb\n")
            with self.assertRaisesRegex(ValueError, "only 2 records"):
                prepare_curve(path, 3)

    def test_latest_checkpoint_ignores_incomplete_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for step in (2, 10):
                checkpoint = root / f"checkpoint-{step}"
                (checkpoint / "adapter").mkdir(parents=True)
                (checkpoint / "state.json").write_text("{}")
                (checkpoint / "training.pt").write_bytes(b"x")
                (checkpoint / "adapter" / "adapter_model.safetensors").write_bytes(b"x")
            (root / "checkpoint-20" / "adapter").mkdir(parents=True)
            self.assertEqual(latest_checkpoint(root), root / "checkpoint-10")


if __name__ == "__main__":
    unittest.main()
