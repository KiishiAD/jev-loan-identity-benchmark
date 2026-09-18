from __future__ import annotations

from dataclasses import asdict, dataclass

from .models import CandidateEvidence, LoanObservation
from .provider import JevResult


@dataclass(frozen=True)
class PolicyConfig:
    min_choice_probability: float = 0.50
    min_choice_margin: float = 0.10
    min_relationship_score: float = 1.50
    min_relationship_confidence: float = 0.35
    min_same_borrower: float = 0.50
    min_same_facility: float = 0.50
    max_material_conflict: float = 0.50

    def to_json_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ResolutionDecision:
    query_id: str
    status: str
    matched_candidate_id: str | None
    reasons: tuple[str, ...]
    model: str
    request_hash: str

    def to_json_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["reasons"] = list(self.reasons)
        return value


class DecisionPolicy:
    """Precision-first policy that keeps deterministic conflicts authoritative."""

    def __init__(self, config: PolicyConfig) -> None:
        self.config = config

    def decide(
        self,
        observation: LoanObservation,
        candidates: list[CandidateEvidence],
        result: JevResult,
    ) -> ResolutionDecision:
        if result.selected_candidate_id is None:
            return ResolutionDecision(
                query_id=observation.query_id,
                status="no_match",
                matched_candidate_id=None,
                reasons=("jev_selected_none",),
                model=result.model,
                request_hash=result.request_hash,
            )

        by_id = {candidate.version.canonical_id: candidate for candidate in candidates}
        selected = by_id.get(result.selected_candidate_id)
        if selected is None:
            raise ValueError("Jev selected a candidate outside the supplied set")
        judgment = result.candidate_judgments.get(result.selected_candidate_id)
        if judgment is None:
            raise ValueError("Jev result omitted judgment for selected candidate")

        candidate_probabilities = [
            probability
            for label, probability in result.choice_probabilities.items()
            if label != "none_or_ambiguous"
        ]
        candidate_probabilities.sort(reverse=True)
        selected_probability = candidate_probabilities[0] if candidate_probabilities else 0.0
        runner_up = max(
            [
                result.choice_probabilities.get("none_or_ambiguous", 0.0),
                *(candidate_probabilities[1:2]),
            ]
        )
        margin = selected_probability - runner_up

        reasons: list[str] = []
        if selected.hard_conflicts:
            reasons.append("deterministic_conflict")
        if selected_probability < self.config.min_choice_probability:
            reasons.append("weak_choice_probability")
        if margin < self.config.min_choice_margin:
            reasons.append("weak_choice_margin")
        if float(judgment["relationship_score"]) < self.config.min_relationship_score:
            reasons.append("weak_relationship")
        if (
            float(judgment["relationship_confidence"])
            < self.config.min_relationship_confidence
        ):
            reasons.append("weak_relationship_confidence")
        if float(judgment["same_borrower"]) < self.config.min_same_borrower:
            reasons.append("weak_same_borrower")
        if float(judgment["same_facility"]) < self.config.min_same_facility:
            reasons.append("weak_same_facility")
        if float(judgment["material_conflict"]) > self.config.max_material_conflict:
            reasons.append("semantic_conflict")

        if reasons:
            return ResolutionDecision(
                query_id=observation.query_id,
                status="review",
                matched_candidate_id=None,
                reasons=tuple(reasons),
                model=result.model,
                request_hash=result.request_hash,
            )
        return ResolutionDecision(
            query_id=observation.query_id,
            status="match",
            matched_candidate_id=result.selected_candidate_id,
            reasons=(),
            model=result.model,
            request_hash=result.request_hash,
        )
