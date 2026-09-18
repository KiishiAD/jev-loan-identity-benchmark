from __future__ import annotations

import hashlib
import json
import sys
import unittest
from collections import Counter
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from loan_identity_jev.synthetic import build_synthetic_fixture
from loan_identity_jev.temporal_store import TemporalLoanStore


class SyntheticFixtureTest(unittest.TestCase):
    def test_generator_is_byte_reproducible_and_grouped_by_borrower(self) -> None:
        first = build_synthetic_fixture(seed=20260918)
        second = build_synthetic_fixture(seed=20260918)

        first_bytes = json.dumps(
            first.to_json_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        second_bytes = json.dumps(
            second.to_json_dict(), sort_keys=True, separators=(",", ":")
        ).encode()

        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(
            hashlib.sha256(first_bytes).hexdigest(),
            hashlib.sha256(second_bytes).hexdigest(),
        )

        split_borrowers: dict[str, set[str]] = {"development": set(), "test": set()}
        versions_by_id = {version.canonical_id: version for version in first.versions}
        for observation in first.observations:
            if observation.expected_canonical_id:
                split_borrowers[observation.split].add(
                    versions_by_id[observation.expected_canonical_id].borrower
                )
        self.assertTrue(split_borrowers["development"])
        self.assertTrue(split_borrowers["test"])
        self.assertTrue(
            split_borrowers["development"].isdisjoint(split_borrowers["test"])
        )

    def test_fixture_has_locked_sizes_and_difficult_negative_slices(self) -> None:
        fixture = build_synthetic_fixture(seed=20260918)
        split_counts = Counter(item.split for item in fixture.observations)
        scenario_counts = Counter(item.scenario for item in fixture.observations)

        self.assertEqual(split_counts, {"development": 30, "test": 60})
        self.assertEqual(len(fixture.versions), 180)
        for scenario in (
            "typo_alias",
            "stale_amendment",
            "same_borrower_sibling",
            "missing_fields",
            "sponsor_confusion_no_match",
            "currency_conflict_no_match",
            "refinancing_no_match",
            "near_name_no_match",
        ):
            self.assertGreater(scenario_counts[scenario], 0)

    def test_every_positive_gold_is_visible_at_its_cutoff(self) -> None:
        fixture = build_synthetic_fixture(seed=20260918)
        store = TemporalLoanStore(fixture.versions)

        for observation in fixture.observations:
            if observation.expected_canonical_id is None:
                continue
            visible = {
                version.canonical_id
                for version in store.as_of(
                    effective_at=observation.effective_at,
                    known_at=observation.known_at,
                )
            }
            self.assertIn(
                observation.expected_canonical_id,
                visible,
                msg=f"gold not visible for {observation.query_id}",
            )


if __name__ == "__main__":
    unittest.main()
