from __future__ import annotations

from datetime import date, datetime
from typing import Iterable

from .models import LoanVersion


class TemporalLoanStore:
    """In-memory bitemporal view over synthetic loan versions.

    ``valid_*`` is business time. ``recorded_*`` is when that version was
    visible to the resolver. This is a generic market-data pattern, not a copy
    of any Bloomberg internal system.
    """

    def __init__(self, versions: Iterable[LoanVersion]) -> None:
        self._versions = tuple(versions)

    def as_of(self, *, effective_at: date, known_at: datetime) -> list[LoanVersion]:
        if known_at.tzinfo is None:
            raise ValueError("known_at must be timezone-aware")
        if effective_at > known_at.date():
            raise ValueError("effective_at cannot be after known_at")

        visible = [
            version
            for version in self._versions
            if version.valid_from <= effective_at
            and (version.valid_to is None or effective_at < version.valid_to)
            and version.recorded_from <= known_at
            and (version.recorded_to is None or known_at < version.recorded_to)
        ]

        by_id: dict[str, LoanVersion] = {}
        for version in visible:
            previous = by_id.get(version.canonical_id)
            if previous is not None:
                raise ValueError(
                    "overlapping bitemporal versions for "
                    f"{version.canonical_id}: {previous.valid_from} and "
                    f"{version.valid_from}"
                )
            by_id[version.canonical_id] = version
        return [by_id[key] for key in sorted(by_id)]
