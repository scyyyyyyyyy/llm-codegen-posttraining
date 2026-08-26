from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data.build_a3prime_pool import matched_control_item, split_item
from train.grpo import RewardFn


class A3PrimeSplitTest(unittest.TestCase):
    def test_split_is_deterministic_and_disjoint(self):
        item = {"id": "mbpp/1", "tests": ["a", "b", "c"]}
        first = split_item(item, visible_count=1, seed=7)
        second = split_item(item, visible_count=1, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first["tests"]), 1)
        self.assertEqual(len(first["heldout_tests"]), 2)
        self.assertEqual(set(first["tests"]) & set(first["heldout_tests"]), set())
        self.assertEqual(set(first["all_tests"]), {"a", "b", "c"})

    def test_items_without_a_holdout_are_excluded(self):
        self.assertIsNone(split_item({"id": "x", "tests": ["a"]}, 1, 0))

    def test_duplicate_assertions_do_not_leak_into_heldout(self):
        split = split_item({"id": "x", "tests": ["a", "a", "b"]}, 1, 0)
        self.assertIsNotNone(split)
        self.assertEqual(len(split["all_tests"]), 2)
        self.assertEqual(len(split["tests"]), 1)
        self.assertEqual(len(split["heldout_tests"]), 1)
        self.assertFalse(set(split["tests"]) & set(split["heldout_tests"]))

    def test_matched_control_uses_same_prompt_and_every_unique_test(self):
        split = split_item({"id": "x", "tests": ["a", "a", "b", "c"]}, 1, 7)
        control = matched_control_item(split)
        self.assertEqual(control["id"], split["id"])
        self.assertEqual(control["tests"], split["all_tests"])
        self.assertEqual(control["heldout_tests"], [])
        self.assertEqual(control["a3prime_condition"], "matched_full_reward_control")


class RewardSubsetTest(unittest.TestCase):
    def test_one_subset_is_shared_by_the_whole_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "curve.jsonl")
            seen = []

            def fake_reward(code, tests):
                seen.append(tuple(tests))
                return float(code[-1])

            reward = RewardFn("partial", 4, path, subsample=1, seed=3)
            with patch("train.grpo.partial_reward", side_effect=fake_reward), patch.object(
                RewardFn, "_pass_fraction", return_value=0.5
            ):
                values = reward(
                    prompts=["p"] * 4,
                    completions=["code0", "code1", "code0", "code1"],
                    tests=[["a", "b", "c"]] * 4,
                )
            self.assertEqual(values, [0.0, 1.0, 0.0, 1.0])
            self.assertEqual(len(set(seen)), 1)
            record = json.loads(Path(path).read_text())
            self.assertIn("hacking_gap", record)
            self.assertEqual(record["n_heldout_rollouts"], 4)

    def test_explicit_visible_split_is_used_for_the_whole_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "curve.jsonl")
            seen = []

            def fake_reward(code, tests):
                seen.append(tuple(tests))
                return float(code[-1])

            reward = RewardFn("partial", 4, path, subsample=1, seed=3)
            with patch("train.grpo.partial_reward", side_effect=fake_reward), patch.object(
                RewardFn, "_pass_fraction", return_value=0.5
            ):
                reward(
                    prompts=["p"] * 4,
                    completions=["code0", "code1", "code0", "code1"],
                    tests=[["visible"]] * 4,
                    heldout_tests=[["heldout-1", "heldout-2"]] * 4,
                )

            self.assertEqual(seen, [("visible",)] * 4)
            record = json.loads(Path(path).read_text())
            self.assertEqual(record["n_heldout_rollouts"], 4)


if __name__ == "__main__":
    unittest.main()
