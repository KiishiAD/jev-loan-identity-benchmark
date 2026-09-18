from __future__ import annotations

import sys
import unittest
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from loan_identity_jev.benchmark import acceptance


class AcceptanceTest(unittest.TestCase):
    def test_any_false_merge_fails_the_precision_first_gate(self) -> None:
        metrics = {
            "confirmed_precision": 0.95,
            "confirmed_recall": 1.0,
            "retrieval_recall_at_k": 1.0,
            "jev_top1_accuracy": 0.98,
            "false_merge_rate": 0.05,
            "latency_seconds": {"p95": 3.0},
            "counts": {"errors": 0, "false_positive": 1},
        }

        result = acceptance(metrics)

        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["zero_false_merges"])


if __name__ == "__main__":
    unittest.main()
