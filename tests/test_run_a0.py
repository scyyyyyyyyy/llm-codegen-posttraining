from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eval.run_a0 import collect


class RunA0MetadataTest(unittest.TestCase):
    def test_collect_preserves_the_requested_arm(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_path = Path(tmp) / "eval_results.json"
            result_path.write_text(json.dumps({
                "eval": {
                    "task/0": [{
                        "solution": "def answer():\n    return 1\n",
                        "base_status": "pass",
                        "plus_status": "pass",
                    }]
                }
            }))
            tasks = {
                "task/0": {
                    "canonical_solution": "return 1",
                    "test": "",
                    "entry_point": "answer",
                }
            }
            with patch("eval.run_a0._load_tasks", return_value=tasks), patch(
                "eval.run_a0.error_breakdown", return_value={"correct": 100.0}
            ), patch(
                "eval.run_a0.stratified_error_breakdown", return_value={}
            ):
                summary = collect(
                    "humaneval",
                    "a3-partial-s2",
                    str(result_path),
                    None,
                    None,
                    tmp,
                    arm="A3",
                )

            self.assertEqual(summary["arm"], "A3")
            saved = json.loads(
                (Path(tmp) / "a0_a3-partial-s2_humaneval.json").read_text()
            )
            self.assertEqual(saved["arm"], "A3")


if __name__ == "__main__":
    unittest.main()
