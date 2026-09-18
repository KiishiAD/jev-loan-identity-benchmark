from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

from .models import LoanObservation
from .policy import ResolutionDecision


@dataclass(frozen=True)
class CaseRecord:
    observation: LoanObservation
    candidate_ids: tuple[str, ...]
    jev_selected_candidate_id: str | None
    decision: ResolutionDecision | None
    end_to_end_latency_seconds: float
    error: str | None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "observation": self.observation.to_json_dict(),
            "candidate_ids": list(self.candidate_ids),
            "jev_selected_candidate_id": self.jev_selected_candidate_id,
            "decision": self.decision.to_json_dict() if self.decision else None,
            "end_to_end_latency_seconds": self.end_to_end_latency_seconds,
            "error": self.error,
        }


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[rank]


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    probability = successes / total
    denominator = 1 + z * z / total
    center = (probability + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            probability * (1 - probability) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _base_metrics(records: list[CaseRecord]) -> dict[str, Any]:
    total = len(records)
    positives = sum(
        record.observation.expected_canonical_id is not None for record in records
    )
    negatives = total - positives
    errors = sum(record.error is not None for record in records)
    matches = [
        record
        for record in records
        if record.decision is not None and record.decision.status == "match"
    ]
    positive_matches = sum(
        record.observation.expected_canonical_id is not None for record in matches
    )
    true_positive = sum(
        record.observation.expected_canonical_id is not None
        and record.decision is not None
        and record.decision.matched_candidate_id
        == record.observation.expected_canonical_id
        for record in matches
    )
    false_positive = len(matches) - true_positive
    false_negative = sum(
        record.observation.expected_canonical_id is not None
        and not (
            record.decision is not None
            and record.decision.status == "match"
            and record.decision.matched_candidate_id
            == record.observation.expected_canonical_id
        )
        for record in records
    )
    true_negative_safe = sum(
        record.observation.expected_canonical_id is None
        and record.decision is not None
        and record.decision.status in {"no_match", "review"}
        for record in records
    )
    reviews = sum(
        record.decision is not None and record.decision.status == "review"
        for record in records
    )
    no_matches = sum(
        record.decision is not None and record.decision.status == "no_match"
        for record in records
    )

    eligible_positive = [
        record
        for record in records
        if record.observation.expected_canonical_id is not None
    ]
    retrieval_at_1 = sum(
        bool(record.candidate_ids)
        and record.candidate_ids[0] == record.observation.expected_canonical_id
        for record in eligible_positive
    )
    retrieval_at_k = sum(
        record.observation.expected_canonical_id in record.candidate_ids
        for record in eligible_positive
    )
    top1_correct = sum(
        record.jev_selected_candidate_id == record.observation.expected_canonical_id
        for record in records
        if record.error is None
    )
    non_error = total - errors
    latencies = [record.end_to_end_latency_seconds for record in records]

    return {
        "counts": {
            "total": total,
            "positives": positives,
            "negatives": negatives,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "safe_negative": true_negative_safe,
            "matches": len(matches),
            "reviews": reviews,
            "no_matches": no_matches,
            "errors": errors,
        },
        "confirmed_precision": _ratio(true_positive, true_positive + false_positive),
        "confirmed_precision_95pct_wilson": wilson_interval(
            true_positive, true_positive + false_positive
        ),
        "confirmed_recall": _ratio(true_positive, positives),
        "confirmed_coverage": _ratio(len(matches), total),
        "positive_coverage": _ratio(positive_matches, positives),
        "false_merge_rate": _ratio(false_positive, len(matches)),
        "review_rate": _ratio(reviews, total),
        "no_match_rate": _ratio(no_matches, total),
        "safe_negative_rate": _ratio(true_negative_safe, negatives),
        "error_rate": _ratio(errors, total),
        "retrieval_recall_at_1": _ratio(retrieval_at_1, positives),
        "retrieval_recall_at_k": _ratio(retrieval_at_k, positives),
        "jev_top1_accuracy": _ratio(top1_correct, non_error),
        "latency_seconds": {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "p99": _percentile(latencies, 0.99),
            "max": max(latencies) if latencies else 0.0,
        },
    }


def compute_metrics(records: Iterable[CaseRecord]) -> dict[str, Any]:
    materialized = list(records)
    metrics = _base_metrics(materialized)
    scenarios = sorted({record.observation.scenario for record in materialized})
    metrics["by_scenario"] = {
        scenario: _base_metrics(
            [
                record
                for record in materialized
                if record.observation.scenario == scenario
            ]
        )
        for scenario in scenarios
    }
    return metrics
