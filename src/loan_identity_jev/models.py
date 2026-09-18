from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any


@dataclass(frozen=True)
class LoanVersion:
    """One bitemporal version of a synthetic canonical loan."""

    canonical_id: str
    borrower: str
    borrower_aliases: tuple[str, ...]
    facility_name: str
    facility_type: str
    lien: str
    currency: str
    amount_millions: str
    maturity: date
    spread_bps: int
    agent: str
    valid_from: date
    valid_to: date | None
    recorded_from: datetime
    recorded_to: datetime | None
    sponsor: str = ""

    def __post_init__(self) -> None:
        if not self.canonical_id.strip():
            raise ValueError("canonical_id is required")
        if self.valid_to is not None and self.valid_to <= self.valid_from:
            raise ValueError("valid_to must be after valid_from")
        if self.recorded_to is not None and self.recorded_to <= self.recorded_from:
            raise ValueError("recorded_to must be after recorded_from")
        if self.recorded_from.tzinfo is None or (
            self.recorded_to is not None and self.recorded_to.tzinfo is None
        ):
            raise ValueError("recorded timestamps must be timezone-aware")
        try:
            amount = Decimal(self.amount_millions)
        except InvalidOperation as error:
            raise ValueError("amount_millions must be a decimal string") from error
        if amount <= 0:
            raise ValueError("amount_millions must be positive")
        if self.spread_bps < 0:
            raise ValueError("spread_bps cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        """Return constructor-compatible values without lossy serialization."""
        return asdict(self)

    def to_json_dict(self) -> dict[str, Any]:
        value = self.to_dict()
        value["borrower_aliases"] = list(self.borrower_aliases)
        for field in ("maturity", "valid_from", "valid_to"):
            item = value[field]
            value[field] = item.isoformat() if item is not None else None
        for field in ("recorded_from", "recorded_to"):
            item = value[field]
            value[field] = item.isoformat() if item is not None else None
        return value

    @classmethod
    def from_json_dict(cls, value: dict[str, Any]) -> "LoanVersion":
        parsed = dict(value)
        parsed["borrower_aliases"] = tuple(parsed["borrower_aliases"])
        for field in ("maturity", "valid_from"):
            parsed[field] = date.fromisoformat(parsed[field])
        if parsed.get("valid_to") is not None:
            parsed["valid_to"] = date.fromisoformat(parsed["valid_to"])
        parsed["recorded_from"] = datetime.fromisoformat(parsed["recorded_from"])
        if parsed.get("recorded_to") is not None:
            parsed["recorded_to"] = datetime.fromisoformat(parsed["recorded_to"])
        return cls(**parsed)


@dataclass(frozen=True)
class LoanObservation:
    """One source record to resolve against the temporal reference store."""

    query_id: str
    split: str
    scenario: str
    expected_canonical_id: str | None
    borrower: str
    facility_name: str
    facility_type: str | None
    lien: str | None
    currency: str | None
    amount_millions: str | None
    maturity: date | None
    spread_bps: int | None
    agent: str | None
    effective_at: date
    known_at: datetime
    sponsor: str | None = None

    def __post_init__(self) -> None:
        if not self.query_id.strip():
            raise ValueError("query_id is required")
        if self.split not in {"development", "test"}:
            raise ValueError("split must be development or test")
        if self.known_at.tzinfo is None:
            raise ValueError("known_at must be timezone-aware")
        if self.effective_at > self.known_at.date():
            raise ValueError("effective_at cannot be after known_at")
        if self.amount_millions is not None:
            try:
                amount = Decimal(self.amount_millions)
            except InvalidOperation as error:
                raise ValueError("amount_millions must be a decimal string") from error
            if amount <= 0:
                raise ValueError("amount_millions must be positive")

    def to_json_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["effective_at"] = self.effective_at.isoformat()
        value["known_at"] = self.known_at.isoformat()
        value["maturity"] = self.maturity.isoformat() if self.maturity else None
        return value

    @classmethod
    def from_json_dict(cls, value: dict[str, Any]) -> "LoanObservation":
        parsed = dict(value)
        parsed["effective_at"] = date.fromisoformat(parsed["effective_at"])
        parsed["known_at"] = datetime.fromisoformat(parsed["known_at"])
        if parsed.get("maturity") is not None:
            parsed["maturity"] = date.fromisoformat(parsed["maturity"])
        return cls(**parsed)


@dataclass(frozen=True)
class CandidateEvidence:
    """A source-backed candidate plus deterministic comparison evidence."""

    version: LoanVersion
    retrieval_score: float
    borrower_similarity: float
    facility_similarity: float
    amount_delta_ratio: float | None
    maturity_delta_days: int | None
    hard_conflicts: tuple[str, ...]

    def to_model_dict(self) -> dict[str, Any]:
        amount_bucket = "missing"
        if self.amount_delta_ratio is not None:
            amount_bucket = (
                "near" if self.amount_delta_ratio <= 0.10 else
                "moderate" if self.amount_delta_ratio <= 0.25 else
                "far"
            )
        maturity_bucket = "missing"
        if self.maturity_delta_days is not None:
            maturity_bucket = (
                "near" if self.maturity_delta_days <= 120 else
                "moderate" if self.maturity_delta_days <= 550 else
                "far"
            )
        return {
            "borrower": self.version.borrower,
            "borrower_aliases": list(self.version.borrower_aliases),
            "sponsor": self.version.sponsor,
            "facility_name": self.version.facility_name,
            "facility_type": self.version.facility_type,
            "lien": self.version.lien,
            "currency": self.version.currency,
            "agent": self.version.agent,
            "amount_comparison": amount_bucket,
            "maturity_comparison": maturity_bucket,
            "deterministic_conflicts": list(self.hard_conflicts),
        }
