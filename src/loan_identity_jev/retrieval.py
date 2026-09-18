from __future__ import annotations

import re
from datetime import date
from decimal import Decimal
from difflib import SequenceMatcher

from .models import CandidateEvidence, LoanObservation, LoanVersion
from .temporal_store import TemporalLoanStore


ABBREVIATIONS = {
    "1l": "first lien",
    "2l": "second lien",
    "tlb": "term loan b",
    "tla": "term loan a",
    "rcf": "revolving credit facility",
    "revolver": "revolving credit facility",
}


def normalize(value: str) -> str:
    text = value.casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens: list[str] = []
    for token in text.split():
        tokens.extend(ABBREVIATIONS.get(token, token).split())
    ignored = {"inc", "llc", "ltd", "limited", "holdings", "corp", "corporation"}
    return " ".join(token for token in tokens if token not in ignored)


def similarity(left: str, right: str) -> float:
    left_normalized, right_normalized = normalize(left), normalize(right)
    if not left_normalized or not right_normalized:
        return 0.0
    left_tokens, right_tokens = set(left_normalized.split()), set(right_normalized.split())
    union = left_tokens | right_tokens
    jaccard = len(left_tokens & right_tokens) / len(union) if union else 0.0
    sequence = SequenceMatcher(None, left_normalized, right_normalized).ratio()
    return max(jaccard, sequence)


def _tranche_label(value: str) -> str | None:
    tokens = normalize(value).split()
    for index, token in enumerate(tokens[:-1]):
        if token == "loan" and len(tokens[index + 1]) == 1:
            label = tokens[index + 1]
            if label.isalpha():
                return label
    return None


def _amount_delta(observation: LoanObservation, candidate: LoanVersion) -> float | None:
    if observation.amount_millions is None:
        return None
    observed = Decimal(observation.amount_millions)
    reference = Decimal(candidate.amount_millions)
    return float(abs(observed - reference) / reference)


def _maturity_delta(observed: date | None, reference: date) -> int | None:
    return abs((observed - reference).days) if observed is not None else None


def _hard_conflicts(
    observation: LoanObservation,
    candidate: LoanVersion,
    amount_delta: float | None,
    maturity_delta: int | None,
) -> tuple[str, ...]:
    conflicts: list[str] = []
    if observation.currency and observation.currency != candidate.currency:
        conflicts.append("currency_conflict")
    if observation.facility_type and observation.facility_type != candidate.facility_type:
        conflicts.append("facility_type_conflict")
    if observation.lien and observation.lien != candidate.lien:
        conflicts.append("lien_conflict")
    observation_tranche = _tranche_label(observation.facility_name)
    candidate_tranche = _tranche_label(candidate.facility_name)
    if observation_tranche is not None and candidate_tranche is None:
        conflicts.append("tranche_unverified")
    elif (
        observation_tranche is not None
        and candidate_tranche is not None
        and observation_tranche != candidate_tranche
    ):
        conflicts.append("tranche_conflict")
    if amount_delta is not None and amount_delta > 0.50:
        conflicts.append("amount_conflict")
    if maturity_delta is not None and maturity_delta > 1_100:
        conflicts.append("maturity_conflict")
    return tuple(conflicts)


class CandidateRetriever:
    def __init__(self, store: TemporalLoanStore, *, top_k: int = 4) -> None:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self._store = store
        self._top_k = top_k

    def retrieve(self, observation: LoanObservation) -> list[CandidateEvidence]:
        candidates: list[CandidateEvidence] = []
        for version in self._store.as_of(
            effective_at=observation.effective_at,
            known_at=observation.known_at,
        ):
            borrower_similarity = max(
                similarity(observation.borrower, name)
                for name in (version.borrower, *version.borrower_aliases)
            )
            facility_similarity = similarity(
                observation.facility_name, version.facility_name
            )
            amount_delta = _amount_delta(observation, version)
            maturity_delta = _maturity_delta(observation.maturity, version.maturity)
            hard_conflicts = _hard_conflicts(
                observation, version, amount_delta, maturity_delta
            )

            amount_score = 0.0
            if amount_delta is not None:
                amount_score = max(0.0, 1.0 - amount_delta)
            score = (
                borrower_similarity * 0.50
                + facility_similarity * 0.25
                + float(observation.facility_type == version.facility_type) * 0.10
                + float(observation.agent == version.agent) * 0.05
                + float(observation.currency == version.currency) * 0.05
                + amount_score * 0.05
                - min(len(hard_conflicts) * 0.08, 0.24)
            )
            candidates.append(
                CandidateEvidence(
                    version=version,
                    retrieval_score=round(score, 6),
                    borrower_similarity=round(borrower_similarity, 6),
                    facility_similarity=round(facility_similarity, 6),
                    amount_delta_ratio=amount_delta,
                    maturity_delta_days=maturity_delta,
                    hard_conflicts=hard_conflicts,
                )
            )
        candidates.sort(
            key=lambda candidate: (
                -candidate.retrieval_score,
                candidate.version.canonical_id,
            )
        )
        return candidates[: self._top_k]