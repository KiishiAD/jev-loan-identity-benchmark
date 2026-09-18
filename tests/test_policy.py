from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from loan_identity_jev.policy import DecisionPolicy, PolicyConfig
from loan_identity_jev.provider import JevResult
from test_provider import build_case


class DecisionPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.observation, self.candidates = build_case()
        selected = self.candidates[0]
        self.strong = JevResult(
            request_hash="hash",
            model="jev-1.13.0",
            selected_candidate_id=selected.version.canonical_id,
            choice_probabilities={
                "option_a": 0.82,
                "option_b": 0.11,
                "none_or_ambiguous": 0.07,
            },
            choice_confidence=0.74,
            candidate_judgments={
                selected.version.canonical_id: {
                    "relationship_score": 1.88,
                    "relationship_probabilities": {
                        "0": 0.02,
                        "1": 0.08,
                        "2": 0.90,
                    },
                    "relationship_confidence": 0.82,
                    "same_borrower": 0.96,
                    "same_facility": 0.94,
                    "material_conflict": 0.03,
                }
            },
            input_tokens=200,
            output_tokens=20,
            live_latency_seconds=0.5,
            execution_mode="live",
        )
        self.policy = DecisionPolicy(PolicyConfig())

    def test_strong_conflict_free_candidate_is_confirmed(self) -> None:
        decision = self.policy.decide(self.observation, self.candidates, self.strong)

        self.assertEqual(decision.status, "match")
        self.assertEqual(
            decision.matched_candidate_id, self.strong.selected_candidate_id
        )
        self.assertEqual(decision.reasons, ())

    def test_deterministic_conflict_blocks_even_strong_jev_answer(self) -> None:
        selected_id = self.strong.selected_candidate_id
        conflicted = [
            replace(
                candidate,
                hard_conflicts=("currency_conflict",),
            )
            if candidate.version.canonical_id == selected_id
            else candidate
            for candidate in self.candidates
        ]

        decision = self.policy.decide(self.observation, conflicted, self.strong)

        self.assertEqual(decision.status, "review")
        self.assertIn("deterministic_conflict", decision.reasons)

    def test_weak_semantic_judgment_routes_to_review(self) -> None:
        selected_id = self.strong.selected_candidate_id
        weak = replace(
            self.strong,
            candidate_judgments={
                selected_id: {
                    **self.strong.candidate_judgments[selected_id],
                    "same_facility": 0.41,
                }
            },
        )

        decision = self.policy.decide(self.observation, self.candidates, weak)

        self.assertEqual(decision.status, "review")
        self.assertIn("weak_same_facility", decision.reasons)

    def test_explicit_none_choice_returns_no_match(self) -> None:
        none = replace(self.strong, selected_candidate_id=None)

        decision = self.policy.decide(self.observation, self.candidates, none)

        self.assertEqual(decision.status, "no_match")
        self.assertIsNone(decision.matched_candidate_id)


if __name__ == "__main__":
    unittest.main()
