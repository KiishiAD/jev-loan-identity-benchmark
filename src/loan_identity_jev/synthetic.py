from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from typing import Any

from .models import LoanObservation, LoanVersion


UTC = timezone.utc
DATASET_VERSION = "synthetic-bitemporal-loans-v1"

BORROWERS = [
    "Northstar Logistics",
    "Blue Lantern Software",
    "Harbor Ridge Dental",
    "Copper Finch Packaging",
    "Redwood Transit",
    "Silver Meridian Health",
    "Stonebridge Data Systems",
    "Summit Peak Education",
    "Cedar Vale Foods",
    "Orchid River Energy",
    "Ironwood Veterinary",
    "Lumen Harbor Media",
    "Pinecrest Security",
    "Aurora Textile Works",
    "Falcon Gate Hospitality",
    "Willow Sphere Telecom",
    "Granite Shore Marine",
    "Emberline Industrial",
    "Clearwater Diagnostics",
    "Juniper Axis Retail",
    "Cobalt Meadow Chemicals",
    "Atlas Grove Services",
    "Mosaic Bay Manufacturing",
    "Keystone Orbit Aerospace",
    "Brightforge Components",
    "Sable Creek Consumer",
    "Velvet Ridge Entertainment",
    "Quantum Orchard Labs",
    "Golden Kite Mobility",
    "Nimbus Terrace Infrastructure",
]
SPONSORS = [
    "Alder Peak Capital",
    "Beacon Hollow Partners",
    "Crescent Field Equity",
    "Driftwood Ridge Capital",
    "Evergreen Arch Partners",
    "Foundry Lake Investments",
]
AGENTS = ["Atlas Bank", "Crown Street Bank", "Meridian Trust", "Union Harbor Bank"]


@dataclass(frozen=True)
class SyntheticFixture:
    versions: tuple[LoanVersion, ...]
    observations: tuple[LoanObservation, ...]
    metadata: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "versions": [version.to_json_dict() for version in self.versions],
            "observations": [item.to_json_dict() for item in self.observations],
        }

    @classmethod
    def from_json_dict(cls, value: dict[str, Any]) -> "SyntheticFixture":
        return cls(
            versions=tuple(
                LoanVersion.from_json_dict(item) for item in value["versions"]
            ),
            observations=tuple(
                LoanObservation.from_json_dict(item)
                for item in value["observations"]
            ),
            metadata=dict(value["metadata"]),
        )


def _at(day: str) -> datetime:
    return datetime.fromisoformat(f"{day}T00:00:00+00:00")


def _opaque_id(seed: int, *parts: str) -> str:
    value = ":".join((DATASET_VERSION, str(seed), *parts))
    return uuid.uuid5(uuid.NAMESPACE_URL, value).hex


def _typo(value: str) -> str:
    words = value.split()
    index = max(range(len(words)), key=lambda item: len(words[item]))
    word = words[index]
    if len(word) > 5:
        pivot = len(word) // 2
        word = word[:pivot] + word[pivot + 1 :]
    words[index] = word
    return " ".join(words)


def _history(
    base: LoanVersion,
    *,
    amended_amount: str,
    amended_maturity: date,
    amended_spread: int,
) -> list[LoanVersion]:
    amendment_effective = date(2025, 1, 1)
    amendment_recorded = _at("2025-01-05")
    initial_open = replace(base, recorded_to=amendment_recorded)
    initial_closed = replace(
        base,
        valid_to=amendment_effective,
        recorded_from=amendment_recorded,
        recorded_to=None,
    )
    amended = replace(
        base,
        amount_millions=amended_amount,
        maturity=amended_maturity,
        spread_bps=amended_spread,
        valid_from=amendment_effective,
        valid_to=None,
        recorded_from=amendment_recorded,
        recorded_to=None,
    )
    return [initial_open, initial_closed, amended]


def _observation(
    *,
    seed: int,
    family: str,
    suffix: str,
    split: str,
    scenario: str,
    expected: str | None,
    source: LoanVersion,
    **overrides: Any,
) -> LoanObservation:
    values: dict[str, Any] = {
        "query_id": _opaque_id(seed, family, suffix),
        "split": split,
        "scenario": scenario,
        "expected_canonical_id": expected,
        "borrower": source.borrower,
        "facility_name": source.facility_name,
        "facility_type": source.facility_type,
        "lien": source.lien,
        "currency": source.currency,
        "amount_millions": source.amount_millions,
        "maturity": source.maturity,
        "spread_bps": source.spread_bps,
        "agent": source.agent,
        "effective_at": date(2025, 7, 1),
        "known_at": _at("2025-08-01"),
        "sponsor": source.sponsor,
    }
    values.update(overrides)
    return LoanObservation(**values)


def build_synthetic_fixture(*, seed: int = 20260918) -> SyntheticFixture:
    """Build a fixed synthetic benchmark with borrower-grouped splits.

    The records imitate a generic bitemporal market-data store. They are not
    Bloomberg data and do not claim to reproduce Bloomberg internals.
    """

    rng = random.Random(seed)
    families = list(BORROWERS)
    rng.shuffle(families)
    versions: list[LoanVersion] = []
    observations: list[LoanObservation] = []

    for position, family in enumerate(families):
        split = "development" if position < 10 else "test"
        sponsor = SPONSORS[position % len(SPONSORS)]
        agent = AGENTS[position % len(AGENTS)]
        currency = ("USD", "EUR", "GBP")[position % 3]
        legal_name = f"{family} Holdings LLC"
        aliases = (family, f"{family.split()[0]} Group")

        term_id = _opaque_id(seed, family, "term")
        term_amount = 320 + position * 13
        term_initial = LoanVersion(
            canonical_id=term_id,
            borrower=legal_name,
            borrower_aliases=aliases,
            facility_name="First Lien Term Loan B",
            facility_type="term_loan",
            lien="first_lien",
            currency=currency,
            amount_millions=str(term_amount),
            maturity=date(2029, 6, 30),
            spread_bps=450,
            agent=agent,
            valid_from=date(2024, 1, 1),
            valid_to=None,
            recorded_from=_at("2024-01-02"),
            recorded_to=None,
            sponsor=sponsor,
        )
        term_history = _history(
            term_initial,
            amended_amount=str(term_amount + 40),
            amended_maturity=date(2030, 6, 30),
            amended_spread=400,
        )
        versions.extend(term_history)
        term_current = term_history[-1]

        sibling_id = _opaque_id(seed, family, "sibling")
        sibling_kind = position % 3
        if sibling_kind == 0:
            sibling_name, sibling_type, sibling_lien = (
                "Revolving Credit Facility",
                "revolver",
                "first_lien",
            )
        elif sibling_kind == 1:
            sibling_name, sibling_type, sibling_lien = (
                "First Lien Term Loan A",
                "term_loan",
                "first_lien",
            )
        else:
            sibling_name, sibling_type, sibling_lien = (
                "Second Lien Term Loan",
                "term_loan",
                "second_lien",
            )
        sibling_amount = 110 + position * 7
        sibling_initial = LoanVersion(
            canonical_id=sibling_id,
            borrower=legal_name,
            borrower_aliases=aliases,
            facility_name=sibling_name,
            facility_type=sibling_type,
            lien=sibling_lien,
            currency=currency,
            amount_millions=str(sibling_amount),
            maturity=date(2028, 12, 31),
            spread_bps=525 if sibling_type == "term_loan" else 325,
            agent=agent,
            valid_from=date(2024, 1, 1),
            valid_to=None,
            recorded_from=_at("2024-01-02"),
            recorded_to=None,
            sponsor=sponsor,
        )
        sibling_history = _history(
            sibling_initial,
            amended_amount=str(sibling_amount + 20),
            amended_maturity=date(2029, 12, 31),
            amended_spread=sibling_initial.spread_bps - 25,
        )
        versions.extend(sibling_history)
        sibling_current = sibling_history[-1]

        positive_a = position % 5
        if positive_a == 0:
            observations.append(
                _observation(
                    seed=seed,
                    family=family,
                    suffix="positive-a",
                    split=split,
                    scenario="typo_alias",
                    expected=term_id,
                    source=term_current,
                    borrower=_typo(family),
                    facility_name="1L TLB",
                )
            )
        elif positive_a == 1:
            observations.append(
                _observation(
                    seed=seed,
                    family=family,
                    suffix="positive-a",
                    split=split,
                    scenario="stale_amendment",
                    expected=term_id,
                    source=term_initial,
                    borrower=family,
                    facility_name="Term Loan B",
                    effective_at=date(2024, 7, 1),
                    known_at=_at("2024-08-01"),
                )
            )
        elif positive_a == 2:
            observations.append(
                _observation(
                    seed=seed,
                    family=family,
                    suffix="positive-a",
                    split=split,
                    scenario="missing_fields",
                    expected=term_id,
                    source=term_current,
                    borrower=family,
                    amount_millions=None,
                    maturity=None,
                    agent=None,
                )
            )
        elif positive_a == 3:
            observations.append(
                _observation(
                    seed=seed,
                    family=family,
                    suffix="positive-a",
                    split=split,
                    scenario="facility_alias",
                    expected=term_id,
                    source=term_current,
                    borrower=aliases[1],
                    facility_name=f"{family.split()[0]} Senior TLB",
                )
            )
        else:
            observations.append(
                _observation(
                    seed=seed,
                    family=family,
                    suffix="positive-a",
                    split=split,
                    scenario="ocr_noise",
                    expected=term_id,
                    source=term_current,
                    borrower=family.replace("i", "1", 1),
                    facility_name="First-Lien  Term\u00a0Loan B",
                )
            )

        positive_b = position % 5
        b_scenarios = (
            "same_borrower_sibling",
            "amendment_current",
            "facility_alias",
            "amount_drift",
            "legal_suffix_variation",
        )
        b_overrides: dict[str, Any] = {"borrower": family}
        if positive_b == 0:
            b_overrides["facility_name"] = (
                "RCF" if sibling_type == "revolver" else sibling_name.replace("Loan", "Ln")
            )
        elif positive_b == 2:
            b_overrides["facility_name"] = (
                "Revolver" if sibling_type == "revolver" else sibling_name.replace("Term Loan", "TL")
            )
        elif positive_b == 3:
            b_overrides["amount_millions"] = str(sibling_amount + 28)
        elif positive_b == 4:
            b_overrides["borrower"] = legal_name.replace(" Holdings LLC", ", L.L.C.")
        observations.append(
            _observation(
                seed=seed,
                family=family,
                suffix="positive-b",
                split=split,
                scenario=b_scenarios[positive_b],
                expected=sibling_id,
                source=sibling_current,
                **b_overrides,
            )
        )

        negative = position % 5
        if negative == 0:
            observations.append(
                _observation(
                    seed=seed,
                    family=family,
                    suffix="negative",
                    split=split,
                    scenario="sponsor_confusion_no_match",
                    expected=None,
                    source=term_current,
                    borrower=sponsor,
                    facility_name="Portfolio Company Acquisition Facility",
                    amount_millions=None,
                    maturity=None,
                    lien=None,
                )
            )
        elif negative == 1:
            other_currency = "EUR" if currency != "EUR" else "USD"
            observations.append(
                _observation(
                    seed=seed,
                    family=family,
                    suffix="negative",
                    split=split,
                    scenario="currency_conflict_no_match",
                    expected=None,
                    source=term_current,
                    borrower=family,
                    currency=other_currency,
                )
            )
        elif negative == 2:
            observations.append(
                _observation(
                    seed=seed,
                    family=family,
                    suffix="negative",
                    split=split,
                    scenario="refinancing_no_match",
                    expected=None,
                    source=term_current,
                    borrower=family,
                    facility_name="2025 Refinancing Term Loan",
                    amount_millions=str(term_amount * 2),
                    maturity=date(2034, 6, 30),
                )
            )
        elif negative == 3:
            observations.append(
                _observation(
                    seed=seed,
                    family=family,
                    suffix="negative",
                    split=split,
                    scenario="near_name_no_match",
                    expected=None,
                    source=term_current,
                    borrower=f"{family} International",
                    facility_name="First Lien Term Loan C",
                    amount_millions=None,
                    maturity=None,
                )
            )
        else:
            observations.append(
                _observation(
                    seed=seed,
                    family=family,
                    suffix="negative",
                    split=split,
                    scenario="unseen_second_lien_no_match",
                    expected=None,
                    source=term_current,
                    borrower=family,
                    facility_name="Second Lien Term Loan C",
                    facility_type="term_loan",
                    lien="second_lien",
                    amount_millions=str(max(75, term_amount // 2)),
                    maturity=date(2032, 6, 30),
                    spread_bps=725,
                )
            )

    return SyntheticFixture(
        versions=tuple(versions),
        observations=tuple(observations),
        metadata={
            "dataset_version": DATASET_VERSION,
            "seed": seed,
            "synthetic_only": True,
            "temporal_model": "bitemporal valid-time plus recorded-time",
            "identity_policy": (
                "Same canonical facility across amendments is a match; different "
                "facility, tranche, lien, currency, refinancing, sponsor, or affiliate "
                "is not a strict match."
            ),
            "development_borrower_families": 10,
            "test_borrower_families": 20,
        },
    )
