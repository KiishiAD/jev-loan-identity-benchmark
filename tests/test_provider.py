from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from loan_identity_jev.models import LoanObservation, LoanVersion
from loan_identity_jev.provider import (
    JevEvaluator,
    JevResult,
    ReplayCache,
    prepare_request,
)
from loan_identity_jev.retrieval import CandidateRetriever
from loan_identity_jev.temporal_store import TemporalLoanStore


def at(day: str) -> datetime:
    return datetime.fromisoformat(f"{day}T00:00:00+00:00")


def build_case() -> tuple[LoanObservation, list]:
    versions = []
    for index, name in enumerate(
        ("Northstar Logistics Holdings LLC", "Northstar Aviation Holdings LLC")
    ):
        versions.append(
            LoanVersion(
                canonical_id=f"opaque-{index}",
                borrower=name,
                borrower_aliases=(name.replace(" Holdings LLC", ""),),
                facility_name="First Lien Term Loan B",
                facility_type="term_loan",
                lien="first_lien",
                currency="USD",
                amount_millions="500",
                maturity=date(2030, 6, 30),
                spread_bps=400,
                agent="Atlas Bank",
                valid_from=date(2025, 1, 1),
                valid_to=None,
                recorded_from=at("2025-01-05"),
                recorded_to=None,
                sponsor="Alder Peak Capital",
            )
        )
    observation = LoanObservation(
        query_id="case-a",
        split="test",
        scenario="hidden-from-model",
        expected_canonical_id="opaque-0",
        borrower="Nortstar Logistics",
        facility_name="1L TLB",
        facility_type="term_loan",
        lien="first_lien",
        currency="USD",
        amount_millions="505",
        maturity=date(2030, 6, 30),
        spread_bps=400,
        agent="Atlas Bank",
        effective_at=date(2025, 6, 30),
        known_at=at("2025-12-31"),
        sponsor="Alder Peak Capital",
    )
    candidates = CandidateRetriever(TemporalLoanStore(versions), top_k=2).retrieve(
        observation
    )
    return observation, candidates


class PreparedRequestTest(unittest.TestCase):
    def test_request_uses_choice_score_and_nouls_without_truth_leakage(self) -> None:
        observation, candidates = build_case()

        prepared = prepare_request(observation, candidates, model="jev-1.13.0")
        serialized = json.dumps(prepared.request_payload(), sort_keys=True)

        self.assertEqual(prepared.questions["best_candidate"].type, "choice")
        for label in prepared.labels:
            self.assertEqual(prepared.questions[f"relationship_{label}"].type, "score")
            self.assertEqual(prepared.questions[f"same_borrower_{label}"].type, "noul")
            self.assertEqual(prepared.questions[f"same_facility_{label}"].type, "noul")
            self.assertEqual(prepared.questions[f"material_conflict_{label}"].type, "noul")
        self.assertNotIn("expected_canonical_id", serialized)
        self.assertNotIn("scenario", serialized)
        self.assertNotIn("split", serialized)
        self.assertNotIn("opaque-0", serialized)
        self.assertNotIn("opaque-1", serialized)

    def test_candidate_permutation_is_deterministic_per_query(self) -> None:
        observation, candidates = build_case()
        first = prepare_request(observation, candidates, model="jev-1.13.0")
        second = prepare_request(observation, candidates, model="jev-1.13.0")

        self.assertEqual(first.candidate_ids, second.candidate_ids)
        self.assertEqual(first.request_hash, second.request_hash)


class FakeAsyncClient:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls = 0

    async def system_one(self, **_: object) -> object:
        self.calls += 1
        return self.response


class JevEvaluatorTest(unittest.TestCase):
    def test_evaluator_maps_opaque_choice_back_to_supplied_candidate(self) -> None:
        observation, candidates = build_case()
        prepared = prepare_request(observation, candidates, model="jev-1.13.0")
        selected_label = prepared.labels[0]
        answers: dict[str, object] = {
            "best_candidate": SimpleNamespace(
                choice=selected_label,
                probabilities={selected_label: 0.91, "none_or_ambiguous": 0.09},
                confidence=0.82,
            )
        }
        for label in prepared.labels:
            answers[f"relationship_{label}"] = SimpleNamespace(
                score=1.9,
                probabilities={"0": 0.02, "1": 0.08, "2": 0.90},
                confidence=0.84,
            )
            answers[f"same_borrower_{label}"] = SimpleNamespace(noul=0.94)
            answers[f"same_facility_{label}"] = SimpleNamespace(noul=0.93)
            answers[f"material_conflict_{label}"] = SimpleNamespace(noul=0.03)
        response = SimpleNamespace(
            model="jev-1.13.0",
            answers=answers,
            usage=SimpleNamespace(input_tokens=321, output_tokens=42),
        )
        client = FakeAsyncClient(response)

        result = asyncio.run(
            JevEvaluator(client=client, model="jev-1.13.0").evaluate(
                observation, candidates
            )
        )

        self.assertEqual(client.calls, 1)
        self.assertEqual(result.selected_candidate_id, prepared.candidate_ids[0])
        self.assertEqual(result.model, "jev-1.13.0")
        self.assertEqual(result.input_tokens, 321)

    def test_replay_cache_round_trips_and_cache_miss_fails(self) -> None:
        result = JevResult(
            request_hash="abc",
            model="jev-1.13.0",
            selected_candidate_id="candidate-x",
            choice_probabilities={"option_a": 0.9, "none_or_ambiguous": 0.1},
            choice_confidence=0.8,
            candidate_judgments={},
            input_tokens=100,
            output_tokens=10,
            live_latency_seconds=0.4,
            execution_mode="live",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "responses.jsonl"
            cache = ReplayCache(path)
            cache.append(result)

            replayed = ReplayCache(path).get("abc")

            self.assertEqual(replayed.selected_candidate_id, "candidate-x")
            self.assertEqual(replayed.execution_mode, "replay")
            with self.assertRaisesRegex(KeyError, "cache miss"):
                ReplayCache(path).get("missing")


if __name__ == "__main__":
    unittest.main()
