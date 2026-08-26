from __future__ import annotations

import json
import unittest
from pathlib import Path

from analysis.a3matched_report import build_summary


class A3MatchedReportTest(unittest.TestCase):
    def test_checked_in_summary_is_reproducible(self):
        root = Path(__file__).resolve().parents[1]
        expected = json.loads(
            (root / "results/a3matched_confirmatory_summary.json").read_text()
        )
        observed = build_summary(root / "results", root / "results")
        self.assertEqual(observed, expected)


if __name__ == "__main__":
    unittest.main()
