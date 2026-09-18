from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Protocol

from typesafe_sdk import Choice, Noul, Score

from .models import CandidateEvidence, LoanObservation


MODEL = "jev-1.13.0"


class AsyncSystemOneClient(Protocol):
    async def system_one(self, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class PreparedRequest:
    state: dict[str, Any]
    questions: dict[str, Choice | Noul | Score]
    labels: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    model: str

    def request_payload(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "questions": {
                key: question.model_dump(exclude_none=True)
                for key, question in self.questions.items()
            },
            "model": self.model,
        }

    @property
    def request_hash(self) -> str:
        payload = json.dumps(
            self.request_payload(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class JevResult:
    request_hash: str
    model: str
    selected_candidate_id: str | None
    choice_probabilities: dict[str, float]
    choice_confidence: float
    candidate_judgments: dict[str, dict[str, Any]]
    input_tokens: int
    output_tokens: int
    live_latency_seconds: float
    execution_mode: str
    replay_latency_seconds: float | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, value: Mapping[str, Any]) -> "JevResult":
        return cls(
            request_hash=str(value["request_hash"]),
            model=str(value["model"]),
            selected_candidate_id=value.get("selected_candidate_id"),
            choice_probabilities={
                str(key): float(probability)
                for key, probability in value["choice_probabilities"].items()
            },
            choice_confidence=float(value["choice_confidence"]),
            candidate_judgments={
                str(key): dict(judgment)
                for key, judgment in value["candidate_judgments"].items()
            },
            input_tokens=int(value["input_tokens"]),
            output_tokens=int(value["output_tokens"]),
            live_latency_seconds=float(value["live_latency_seconds"]),
            execution_mode=str(value["execution_mode"]),
            replay_latency_seconds=(
                float(value["replay_latency_seconds"])
                if value.get("replay_latency_seconds") is not None
                else None
            ),
        )


def _permuted_candidates(
    observation: LoanObservation, candidates: list[CandidateEvidence]
) -> list[CandidateEvidence]:
    return sorted(
        candidates,
        key=lambda candidate: hashlib.sha256(
            f"{observation.query_id}:{candidate.version.canonical_id}".encode()
        ).digest(),
    )


def prepare_request(
    observation: LoanObservation,
    candidates: list[CandidateEvidence],
    *,
    model: str = MODEL,
) -> PreparedRequest:
    ordered = _permuted_candidates(observation, candidates)
    labels = tuple(f"option_{chr(ord('a') + index)}" for index in range(len(ordered)))
    candidate_ids = tuple(candidate.version.canonical_id for candidate in ordered)
    state = {
        "observation": {
            "borrower": observation.borrower,
            "facility_name": observation.facility_name,
            "facility_type": observation.facility_type,
            "lien": observation.lien,
            "currency": observation.currency,
            "agent": observation.agent,
            "sponsor": observation.sponsor,
        },
        "candidates": [candidate.to_model_dict() for candidate in ordered],
        "identity_policy": {
            "same": (
                "The same strict loan facility across spelling changes, aliases, and "
                "ordinary amendments that preserve facility identity."
            ),
            "different": (
                "A different borrower, facility, tranche, lien, currency, refinancing, "
                "or merely related sponsor/affiliate is a different identity."
            ),
            "uncertain": "Insufficient or contradictory evidence requires review.",
        },
    }

    choice_criteria = {
        label: (
            f"`candidates[{index}]` is the same strict loan facility as `observation`, "
            "not merely a related borrower or sibling facility."
        )
        for index, label in enumerate(labels)
    }
    choice_criteria["none_or_ambiguous"] = (
        "No candidate is a safe strict identity match, or the evidence is ambiguous."
    )
    questions: dict[str, Choice | Noul | Score] = {
        "best_candidate": Choice(
            instructions=(
                "Which supplied candidate, if any, is the same strict loan facility as "
                "`observation` under `identity_policy`?"
            ),
            criteria=choice_criteria,
        )
    }
    relationship_levels = [
        "Different loan identity: different borrower, facility, tranche, lien, currency, refinancing, sponsor, or affiliate.",
        "Related or plausible but insufficient or contradictory evidence; a human must review.",
        "The same strict loan facility despite harmless name variation or a valid amendment in its lineage.",
    ]
    for index, label in enumerate(labels):
        candidate_path = f"`candidates[{index}]`"
        questions[f"relationship_{label}"] = Score(
            instructions=(
                f"How does {candidate_path} relate to `observation` as a strict loan identity?"
            ),
            criteria=relationship_levels,
        )
        questions[f"same_borrower_{label}"] = Noul(
            instructions=(
                f"Do {candidate_path} and `observation` refer to the same legal borrower, "
                "allowing stated aliases but not a sponsor, parent, or similar affiliate?"
            ),
            criteria={
                "true": "Same legal borrower or an explicitly listed alias.",
                "false": "Different, merely related, sponsor, parent, affiliate, or unclear.",
            },
        )
        questions[f"same_facility_{label}"] = Noul(
            instructions=(
                f"Do {candidate_path} and `observation` describe the same exact facility or "
                "tranche, rather than another facility of the same borrower?"
            ),
            criteria={
                "true": "Same facility/tranche despite harmless wording or amendment.",
                "false": "Different facility, tranche, lien, currency, refinancing, or unclear.",
            },
        )
        questions[f"material_conflict_{label}"] = Noul(
            instructions=(
                f"Is there an identity-changing conflict between {candidate_path} and "
                "`observation`, including borrower, facility, tranche, lien, currency, "
                "refinancing, or sponsor-versus-borrower confusion?"
            ),
            criteria={
                "true": "At least one material identity conflict is present.",
                "false": "No material identity conflict is present.",
            },
        )

    return PreparedRequest(
        state=state,
        questions=questions,
        labels=labels,
        candidate_ids=candidate_ids,
        model=model,
    )


class JevEvaluator:
    def __init__(self, *, client: AsyncSystemOneClient, model: str = MODEL) -> None:
        self._client = client
        self._model = model

    async def evaluate(
        self,
        observation: LoanObservation,
        candidates: list[CandidateEvidence],
    ) -> JevResult:
        if not candidates:
            raise ValueError("Jev evaluation requires at least one candidate")
        prepared = prepare_request(observation, candidates, model=self._model)
        started = time.perf_counter()
        response = await self._client.system_one(
            state=prepared.state,
            questions=prepared.questions,
            model=self._model,
        )
        latency = time.perf_counter() - started
        choice = response.answers["best_candidate"]
        selected_label = choice.choice
        selected_candidate_id = None
        if selected_label != "none_or_ambiguous":
            selected_candidate_id = prepared.candidate_ids[
                prepared.labels.index(selected_label)
            ]

        judgments: dict[str, dict[str, Any]] = {}
        for label, candidate_id in zip(prepared.labels, prepared.candidate_ids):
            relationship = response.answers[f"relationship_{label}"]
            judgments[candidate_id] = {
                "relationship_score": float(relationship.score),
                "relationship_probabilities": {
                    str(key): float(value)
                    for key, value in relationship.probabilities.items()
                },
                "relationship_confidence": float(relationship.confidence),
                "same_borrower": float(
                    response.answers[f"same_borrower_{label}"].noul
                ),
                "same_facility": float(
                    response.answers[f"same_facility_{label}"].noul
                ),
                "material_conflict": float(
                    response.answers[f"material_conflict_{label}"].noul
                ),
            }
        usage = response.usage
        return JevResult(
            request_hash=prepared.request_hash,
            model=str(response.model),
            selected_candidate_id=selected_candidate_id,
            choice_probabilities={
                str(key): float(value) for key, value in choice.probabilities.items()
            },
            choice_confidence=float(choice.confidence),
            candidate_judgments=judgments,
            input_tokens=int(usage.input_tokens or 0),
            output_tokens=int(usage.output_tokens or 0),
            live_latency_seconds=latency,
            execution_mode="live",
        )


class ReplayCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._results: dict[str, JevResult] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                result = JevResult.from_json_dict(json.loads(line))
                self._results[result.request_hash] = result

    def append(self, result: JevResult) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result.to_json_dict(), sort_keys=True) + "\n")
        self._results[result.request_hash] = result

    def get(self, request_hash: str) -> JevResult:
        started = time.perf_counter()
        if request_hash not in self._results:
            raise KeyError(f"replay cache miss for request {request_hash}")
        result = self._results[request_hash]
        return replace(
            result,
            execution_mode="replay",
            replay_latency_seconds=time.perf_counter() - started,
        )
