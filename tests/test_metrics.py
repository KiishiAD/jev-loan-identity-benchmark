from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from loan_identity_jev.metrics import CaseRecord, compute_metrics
from loan_identity_jev.policy import ResolutionDecision
from test_provider import build_case


class MetricsTest(unittest.TestCase):
    def test_precision_recall_coverage_and_latency_are_explicit(self) -> None:
        base_observation, _ = build_case()

        def observation(number: int, expected: str | None):
            return replace(
                base_observation,
                query_id=f"q-{number}",
                expected_canonical_id=expected,
                scenario="scenario-a" if number < 4 else "scenario-b",
            )

        def decision(number: int, status: str, matched: str | None):
            return ResolutionDecision(
                query_id=f"q-{number}",
                status=status,
                matched_candidate_id=matched,
                reasons=(),
                model="jev-1.13.0",
                request_hash=f"h-{number}",
            )

        cases = [
            CaseRecord(observation(1, "a"), ("a", "x"), "a", decision(1, "match", "a"), 0.10, None),
            CaseRecord(observation(2, "b"), ("b", "x"), "b", decision(2, "match", "b"), 0.20, None),
            CaseRecord(observation(3, "c"), ("x", "c"), "x", decision(3, "review", None), 0.30, None),
            CaseRecord(observation(4, None), ("x", "y"), None, decision(4, "no_match", None), 0.40, None),
            CaseRecord(observation(5, None), ("z", "y"), "z", decision(5, "match", "z"), 0.50, None),
        ]

        metrics = compute_metrics(cases)

        self.assertEqual(metrics["counts"]["true_positive"], 2)
        self.assertEqual(metrics["counts"]["false_positive"], 1)
        self.assertEqual(metrics["counts"]["false_negative"], 1)
        self.assertAlmostEqual(metrics["confirmed_precision"], 2 / 3)
        self.assertAlmostEqual(metrics["confirmed_recall"], 2 / 3)
        self.assertAlmostEqual(metrics["confirmed_coverage"], 3 / 5)
        self.assertAlmostEqual(metrics["positive_coverage"], 2 / 3)
        self.assertAlmostEqual(metrics["review_rate"], 1 / 5)
        self.assertAlmostEqual(metrics["no_match_rate"], 1 / 5)
        self.assertAlmostEqual(metrics["retrieval_recall_at_k"], 1.0)
        self.assertAlmostEqual(metrics["retrieval_recall_at_1"], 2 / 3)
        self.assertAlmostEqual(metrics["latency_seconds"]["p50"], 0.30)
        self.assertAlmostEqual(metrics["latency_seconds"]["p95"], 0.50)

    def test_provider_errors_stay_in_denominator(self) -> None:
        observation, _ = build_case()
        case = CaseRecord(
            observation=observation,
            candidate_ids=(),
            jev_selected_candidate_id=None,
            decision=None,
            end_to_end_latency_seconds=10.0,
            error="provider unavailable",
        )

        metrics = compute_metrics([case])

        self.assertEqual(metrics["counts"]["errors"], 1)
        self.assertEqual(metrics["error_rate"], 1.0)
        self.assertEqual(metrics["confirmed_coverage"], 0.0)


if __name__ == "__main__":
    unittest.main()
