from __future__ import annotations

import sys
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from loan_identity_jev.models import LoanVersion
from loan_identity_jev.temporal_store import TemporalLoanStore


UTC = timezone.utc


def at(day: str) -> datetime:
    return datetime.fromisoformat(f"{day}T00:00:00+00:00")


class TemporalLoanStoreTest(unittest.TestCase):
    def test_as_of_query_respects_business_and_recorded_time(self) -> None:
        original_open = LoanVersion(
            canonical_id="LN-001",
            borrower="Northstar Logistics Holdings LLC",
            borrower_aliases=("Northstar Logistics",),
            facility_name="First Lien Term Loan B",
            facility_type="term_loan",
            lien="first_lien",
            currency="USD",
            amount_millions="500",
            maturity=date(2029, 6, 30),
            spread_bps=450,
            agent="Atlas Bank",
            valid_from=date(2024, 1, 1),
            valid_to=None,
            recorded_from=at("2024-01-02"),
            recorded_to=at("2025-01-05"),
        )
        original_closed = LoanVersion(
            **{
                **original_open.to_dict(),
                "valid_to": date(2025, 1, 1),
                "recorded_from": at("2025-01-05"),
                "recorded_to": None,
            }
        )
        amended = LoanVersion(
            **{
                **original_open.to_dict(),
                "amount_millions": "550",
                "maturity": date(2030, 6, 30),
                "spread_bps": 400,
                "valid_from": date(2025, 1, 1),
                "valid_to": None,
                "recorded_from": at("2025-01-05"),
                "recorded_to": None,
            }
        )
        store = TemporalLoanStore([original_open, original_closed, amended])

        before_amendment_was_known = store.as_of(
            effective_at=date(2024, 6, 30),
            known_at=at("2024-12-31"),
        )
        after_amendment_was_known = store.as_of(
            effective_at=date(2025, 6, 30),
            known_at=at("2025-12-31"),
        )

        self.assertEqual(before_amendment_was_known[0].amount_millions, "500")
        self.assertEqual(before_amendment_was_known[0].maturity, date(2029, 6, 30))
        self.assertEqual(after_amendment_was_known[0].amount_millions, "550")
        self.assertEqual(after_amendment_was_known[0].maturity, date(2030, 6, 30))

    def test_rejects_effective_time_after_information_cutoff(self) -> None:
        store = TemporalLoanStore([])

        with self.assertRaisesRegex(ValueError, "effective_at cannot be after known_at"):
            store.as_of(
                effective_at=date(2025, 1, 2),
                known_at=at("2025-01-01"),
            )


if __name__ == "__main__":
    unittest.main()
