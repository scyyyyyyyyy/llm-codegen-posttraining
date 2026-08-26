from __future__ import annotations

import unittest

from analysis.a4_report import linear_slope, summarize_curve


class A4ReportTest(unittest.TestCase):
    def test_curve_summary_records_direction_and_budget(self):
        records = []
        for step in range(1, 99):
            records.append(
                {
                    "optimizer_step": step,
                    "rollouts_completed": min(3136, step * 32),
                    "rollouts_this_update": 32,
                    "cumulative_completion_tokens": step * 100,
                    "sampled_reverse_kl_per_token": 1.0 - step / 100.0,
                    "elapsed_seconds": step * 2.0,
                    "max_gpu_memory_allocated_bytes": step,
                }
            )
        records[-1]["rollouts_completed"] = 3136
        summary = summarize_curve(records)
        self.assertEqual(summary["records"], 98)
        self.assertEqual(summary["rollouts"], 3136)
        self.assertLess(summary["last_minus_first_10_mean_sampled_kl"], 0)
        self.assertLess(summary["linear_slope_per_update"], 0)

    def test_linear_slope(self):
        self.assertAlmostEqual(linear_slope([1.0, 3.0, 5.0]), 2.0)
        with self.assertRaisesRegex(ValueError, "at least two"):
            linear_slope([1.0])


if __name__ == "__main__":
    unittest.main()
