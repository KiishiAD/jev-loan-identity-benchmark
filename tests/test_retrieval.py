from __future__ import annotations

import sys
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from loan_identity_jev.models import LoanObservation, LoanVersion
from loan_identity_jev.retrieval import CandidateRetriever
from loan_identity_jev.temporal_store import TemporalLoanStore


UTC = timezone.utc


def at(day: str) -> datetime:
    return datetime.fromisoformat(f"{day}T00:00:00+00:00")


def version(
    canonical_id: str,
    borrower: str,
    facility_name: str,
    facility_type: str,
    lien: str,
    currency: str = "USD",
) -> LoanVersion:
    return LoanVersion(
        canonical_id=canonical_id,
        borrower=borrower,
        borrower_aliases=(borrower.replace(" Holdings LLC", ""),),
        facility_name=facility_name,
        facility_type=facility_type,
        lien=lien,
        currency=currency,
        amount_millions="500",
        maturity=date(2030, 6, 30),
        spread_bps=400,
        agent="Atlas Bank",
        valid_from=date(2025, 1, 1),
        valid_to=None,
        recorded_from=at("2025-01-05"),
        recorded_to=None,
    )


class CandidateRetrieverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.store = TemporalLoanStore(
            [
                version(
                    "LN-001",
                    "Northstar Logistics Holdings LLC",
                    "First Lien Term Loan B",
                    "term_loan",
                    "first_lien",
                ),
                version(
                    "LN-002",
                    "Northstar Logistics Holdings LLC",
                    "Revolving Credit Facility",
                    "revolver",
                    "first_lien",
                ),
                version(
                    "LN-003",
                    "Blue Lantern Software Inc",
                    "First Lien Term Loan",
                    "term_loan",
                    "first_lien",
                    "EUR",
                ),
                version(
                    "LN-004",
                    "Northstar Logistics Holdings LLC",
                    "Second Lien Term Loan",
                    "term_loan",
                    "second_lien",
                ),
            ]
        )
        self.retriever = CandidateRetriever(self.store, top_k=3)

    def test_typo_and_abbreviation_still_retrieve_correct_loan_first(self) -> None:
        observation = LoanObservation(
            query_id="Q-001",
            split="test",
            scenario="typo_alias",
            expected_canonical_id="LN-001",
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
        )

        candidates = self.retriever.retrieve(observation)

        self.assertEqual(candidates[0].version.canonical_id, "LN-001")
        self.assertGreater(candidates[0].retrieval_score, candidates[1].retrieval_score)
        self.assertEqual(candidates[0].hard_conflicts, ())

    def test_sibling_facility_is_kept_but_conflicts_are_explicit(self) -> None:
        observation = LoanObservation(
            query_id="Q-002",
            split="test",
            scenario="unseen_second_lien",
            expected_canonical_id=None,
            borrower="Northstar Logistics",
            facility_name="Second Lien Term Loan C",
            facility_type="term_loan",
            lien="second_lien",
            currency="USD",
            amount_millions="300",
            maturity=date(2031, 6, 30),
            spread_bps=700,
            agent="Atlas Bank",
            effective_at=date(2025, 6, 30),
            known_at=at("2025-12-31"),
        )

        candidates = self.retriever.retrieve(observation)
        by_id = {candidate.version.canonical_id: candidate for candidate in candidates}

        self.assertIn("LN-001", by_id)
        self.assertIn("lien_conflict", by_id["LN-001"].hard_conflicts)
        self.assertIn("facility_type_conflict", by_id["LN-002"].hard_conflicts)
        self.assertIn("tranche_unverified", by_id["LN-004"].hard_conflicts)

    def test_versions_not_known_at_cutoff_are_not_candidates(self) -> None:
        observation = LoanObservation(
            query_id="Q-003",
            split="test",
            scenario="cutoff",
            expected_canonical_id=None,
            borrower="Northstar Logistics",
            facility_name="First Lien Term Loan B",
            facility_type="term_loan",
            lien="first_lien",
            currency="USD",
            amount_millions="500",
            maturity=date(2030, 6, 30),
            spread_bps=400,
            agent="Atlas Bank",
            effective_at=date(2025, 1, 3),
            known_at=at("2025-01-04"),
        )

        self.assertEqual(self.retriever.retrieve(observation), [])


if __name__ == "__main__":
    unittest.main()
