from __future__ import annotations

import argparse
import asyncio
import hashlib
import itertools
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from typesafe_sdk import AsyncTypeSafeClient

from .metrics import CaseRecord, compute_metrics
from .models import CandidateEvidence, LoanObservation
from .policy import DecisionPolicy, PolicyConfig, ResolutionDecision
from .provider import JevEvaluator, JevResult, MODEL, ReplayCache, prepare_request
from .retrieval import CandidateRetriever
from .synthetic import SyntheticFixture, build_synthetic_fixture
from .temporal_store import TemporalLoanStore


@dataclass(frozen=True)
class RawCase:
    observation: LoanObservation
    candidates: tuple[CandidateEvidence, ...]
    result: JevResult | None
    end_to_end_latency_seconds: float
    error: str | None


@dataclass(frozen=True)
class RunSummary:
    report_path: Path
    run_dir: Path
    acceptance_passed: bool
    metrics: dict[str, Any]


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True) + "\n")


def _git_commit(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


async def _score_live(
    observations: list[LoanObservation],
    retriever: CandidateRetriever,
    evaluator: JevEvaluator,
    cache: ReplayCache,
    *,
    concurrency: int,
) -> tuple[list[RawCase], float]:
    semaphore = asyncio.Semaphore(concurrency)

    async def score_one(observation: LoanObservation) -> RawCase:
        started = time.perf_counter()
        candidates = tuple(retriever.retrieve(observation))
        if not candidates:
            return RawCase(
                observation=observation,
                candidates=(),
                result=None,
                end_to_end_latency_seconds=time.perf_counter() - started,
                error=None,
            )
        try:
            async with semaphore:
                result = await evaluator.evaluate(observation, list(candidates))
            cache.append(result)
            return RawCase(
                observation=observation,
                candidates=candidates,
                result=result,
                end_to_end_latency_seconds=time.perf_counter() - started,
                error=None,
            )
        except Exception as error:  # provider boundary: preserve row, fail closed
            return RawCase(
                observation=observation,
                candidates=candidates,
                result=None,
                end_to_end_latency_seconds=time.perf_counter() - started,
                error=f"provider_error:{type(error).__name__}",
            )

    started = time.perf_counter()
    records = await asyncio.gather(*(score_one(item) for item in observations))
    return records, time.perf_counter() - started


def _decision_for_case(raw: RawCase, policy: DecisionPolicy) -> ResolutionDecision | None:
    if raw.error is not None:
        return None
    if not raw.candidates:
        return ResolutionDecision(
            query_id=raw.observation.query_id,
            status="no_match",
            matched_candidate_id=None,
            reasons=("no_temporally_eligible_candidates",),
            model="deterministic",
            request_hash="",
        )
    if raw.result is None:
        return None
    return policy.decide(raw.observation, list(raw.candidates), raw.result)


def apply_policy(raw_cases: Iterable[RawCase], config: PolicyConfig) -> list[CaseRecord]:
    policy = DecisionPolicy(config)
    records: list[CaseRecord] = []
    for raw in raw_cases:
        decision = _decision_for_case(raw, policy)
        records.append(
            CaseRecord(
                observation=raw.observation,
                candidate_ids=tuple(
                    candidate.version.canonical_id for candidate in raw.candidates
                ),
                jev_selected_candidate_id=(
                    raw.result.selected_candidate_id if raw.result else None
                ),
                decision=decision,
                end_to_end_latency_seconds=raw.end_to_end_latency_seconds,
                error=raw.error,
            )
        )
    return records


def tune_policy(raw_cases: list[RawCase]) -> tuple[PolicyConfig, dict[str, Any]]:
    """Choose a precision-first policy using development rows only."""

    choices = {
        "min_choice_probability": (0.35, 0.45, 0.55, 0.65),
        "min_choice_margin": (0.00, 0.05, 0.10, 0.20),
        "min_relationship_score": (1.20, 1.35, 1.50, 1.65),
        "min_relationship_confidence": (0.00, 0.20, 0.35, 0.50),
        "min_same_borrower": (0.35, 0.50, 0.65),
        "min_same_facility": (0.35, 0.50, 0.65),
        "max_material_conflict": (0.35, 0.50, 0.65),
    }
    names = tuple(choices)
    candidates: list[tuple[tuple[float, ...], PolicyConfig, dict[str, Any]]] = []
    for values in itertools.product(*(choices[name] for name in names)):
        config = PolicyConfig(**dict(zip(names, values)))
        metrics = compute_metrics(apply_policy(raw_cases, config))
        precision = metrics["confirmed_precision"]
        recall = metrics["confirmed_recall"]
        false_positives = metrics["counts"]["false_positive"]
        if precision < 0.95 or false_positives > 0:
            continue
        rank = (
            recall,
            metrics["confirmed_coverage"],
            metrics["jev_top1_accuracy"],
            precision,
        )
        candidates.append((rank, config, metrics))
    if not candidates:
        fallback = PolicyConfig()
        return fallback, compute_metrics(apply_policy(raw_cases, fallback))
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, config, metrics = candidates[0]
    return config, metrics


def _replay_cases(
    observations: list[LoanObservation],
    retriever: CandidateRetriever,
    cache: ReplayCache,
    *,
    model: str,
) -> list[RawCase]:
    records: list[RawCase] = []
    for observation in observations:
        started = time.perf_counter()
        candidates = tuple(retriever.retrieve(observation))
        if not candidates:
            records.append(
                RawCase(
                    observation=observation,
                    candidates=(),
                    result=None,
                    end_to_end_latency_seconds=time.perf_counter() - started,
                    error=None,
                )
            )
            continue
        prepared = prepare_request(observation, list(candidates), model=model)
        result = cache.get(prepared.request_hash)
        records.append(
            RawCase(
                observation=observation,
                candidates=candidates,
                result=result,
                end_to_end_latency_seconds=time.perf_counter() - started,
                error=None,
            )
        )
    return records


def _quality_projection(metrics: dict[str, Any]) -> dict[str, Any]:
    value = dict(metrics)
    value.pop("latency_seconds", None)
    value["by_scenario"] = {
        scenario: _quality_projection(scenario_metrics)
        for scenario, scenario_metrics in value.get("by_scenario", {}).items()
    }
    return value


def acceptance(metrics: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "confirmed_precision_at_least_90pct": metrics["confirmed_precision"] >= 0.90,
        "confirmed_recall_at_least_80pct": metrics["confirmed_recall"] >= 0.80,
        "retrieval_recall_at_k_at_least_98pct": metrics["retrieval_recall_at_k"] >= 0.98,
        "jev_top1_accuracy_at_least_90pct": metrics["jev_top1_accuracy"] >= 0.90,
        "false_merge_rate_at_most_10pct": metrics["false_merge_rate"] <= 0.10,
        "zero_false_merges": metrics["counts"]["false_positive"] == 0,
        "live_p95_at_most_10_seconds": metrics["latency_seconds"]["p95"] <= 10.0,
        "zero_provider_errors": metrics["counts"]["errors"] == 0,
    }
    return {"passed": all(checks.values()), "checks": checks}


def _render_report(
    *,
    report_date: str,
    git_commit: str,
    fixture: SyntheticFixture,
    dataset_hash: str,
    policy: PolicyConfig,
    policy_hash: str,
    development_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    replay_metrics: dict[str, Any],
    acceptance_result: dict[str, Any],
    live_wall_seconds: float,
    live_response_hash: str,
    replay_identical: bool,
    run_dir_name: str,
) -> str:
    counts = test_metrics["counts"]
    latency = test_metrics["latency_seconds"]
    precision_low, precision_high = test_metrics[
        "confirmed_precision_95pct_wilson"
    ]
    check_lines = "\n".join(
        f"- {'PASS' if passed else 'FAIL'}: `{name}`"
        for name, passed in acceptance_result["checks"].items()
    )
    scenario_lines = []
    for scenario, values in test_metrics["by_scenario"].items():
        scenario_lines.append(
            "- `{}`: n={}, precision={:.1%}, recall={:.1%}, matches={}, FP={}".format(
                scenario,
                values["counts"]["total"],
                values["confirmed_precision"],
                values["confirmed_recall"],
                values["counts"]["matches"],
                values["counts"]["false_positive"],
            )
        )
    return f"""# Jev semantic loan identity benchmark report — {report_date}

- **Decision:** {'PASS' if acceptance_result['passed'] else 'FAIL'}
- **Execution date:** {report_date} UTC
- **Benchmark code commit:** `{git_commit}`
**Jev model pinned:** `{MODEL}`

## Scope and evidence boundary

This is a **synthetic** semantic identity benchmark. It simulates a generic bitemporal
market/reference-data store with valid time and recorded time. It does **not** use Bloomberg,
employer, client, licensed, or production loan data, and it does not claim to reproduce
Bloomberg's internal architecture. The direct evidence below applies only to this generated
workload and this pinned model/question/policy configuration.

Jev is used after deterministic temporal filtering and candidate retrieval. It receives
only supplied candidates and makes bounded `Choice`, `Score`, and `Noul` judgments. Code
owns dates, amounts, hard conflicts, identifiers, thresholds, and abstention.

## Dataset

- Version: `{fixture.metadata['dataset_version']}`
- Seed: `{fixture.metadata['seed']}`
- Bitemporal version rows: {len(fixture.versions)}
- Development observations: 30, grouped into 10 borrower families
- Held-out test observations: {counts['total']}, grouped into 20 disjoint borrower families
- Test positives: {counts['positives']}; hard negatives/no-match: {counts['negatives']}
- Dataset SHA-256: `{dataset_hash}`
- Identity rule: {fixture.metadata['identity_policy']}

Cases include typos, abbreviations, legal-suffix variants, old amendment terms, missing
fields, same-borrower sibling facilities/tranches/liens, sponsor-versus-borrower confusion,
currency conflicts, refinancing, near-name affiliates, OCR-like corruption, and unseen
second-lien facilities.

## Held-out live results

- Retrieval Recall@1: **{test_metrics['retrieval_recall_at_1']:.2%}**
- Retrieval Recall@4: **{test_metrics['retrieval_recall_at_k']:.2%}**
- Jev raw top-1 accuracy, including no-match: **{test_metrics['jev_top1_accuracy']:.2%}**
- Confirmed precision: **{test_metrics['confirmed_precision']:.2%}** ({counts['true_positive']}/{counts['true_positive'] + counts['false_positive']}; 95% Wilson interval {precision_low:.2%}–{precision_high:.2%})
- Confirmed recall: **{test_metrics['confirmed_recall']:.2%}** ({counts['true_positive']}/{counts['positives']})
- Confirmed coverage: **{test_metrics['confirmed_coverage']:.2%}** ({counts['matches']}/{counts['total']})
- False merges: **{counts['false_positive']}**; false-merge rate **{test_metrics['false_merge_rate']:.2%}**
- Review rate: **{test_metrics['review_rate']:.2%}**; no-match rate: **{test_metrics['no_match_rate']:.2%}**
- Provider errors: **{counts['errors']}**
- Live latency: p50 **{latency['p50']:.3f}s**, p95 **{latency['p95']:.3f}s**, p99 **{latency['p99']:.3f}s**, max **{latency['max']:.3f}s**
- Live scored wall time: **{live_wall_seconds:.3f}s** for 90 development + test calls
- Throughput at concurrency 6: **{90 / live_wall_seconds:.2f} observations/s**

### Scenario slices

{chr(10).join(scenario_lines)}

## Development-only policy selection

The decision thresholds were selected on the 30-row development split before evaluating
the 60-row borrower-disjoint test split. The test split was not used by `tune_policy`.
Development confirmed precision was {development_metrics['confirmed_precision']:.2%} and
confirmed recall was {development_metrics['confirmed_recall']:.2%}.

```json
{json.dumps(policy.to_json_dict(), indent=2, sort_keys=True)}
```

Policy SHA-256: `{policy_hash}`

## Acceptance gates

{check_lines}

## Replay proof

The saved sanitized response cache was replayed with no API calls. Decisions and all quality
metrics were {'byte-equivalent after excluding timing fields' if replay_identical else 'NOT equivalent'}.
Replay p50/p95 were {replay_metrics['latency_seconds']['p50']:.6f}s / {replay_metrics['latency_seconds']['p95']:.6f}s.
Live response-cache SHA-256: `{live_response_hash}`.

## Reproduce

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .

# Offline: tests and exact saved-result replay need no key.
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m loan_identity_jev.benchmark replay \\
  --run-dir reports/{run_dir_name}

# Live rerun: synthetic data only. This spends TypeSafe credits.
export TYPESAFE_API_KEY=...  # never commit it
.venv/bin/python -m loan_identity_jev.benchmark live \\
  --repo-root . --report-date {report_date} --overwrite
```

A live rerun can differ because hosted model/service behavior can change. The benchmark pins
`{MODEL}` and records response and configuration hashes so drift is visible.

## Artifacts

- `reports/{run_dir_name}/dataset.json`
- `reports/{run_dir_name}/policy-config.json`
- `reports/{run_dir_name}/live-responses.jsonl`
- `reports/{run_dir_name}/live-outcomes.jsonl`
- `reports/{run_dir_name}/replay-outcomes.jsonl`
- `reports/{run_dir_name}/metrics.json`
- `reports/{run_dir_name}/manifest.json`

## Limits

- This is 60 held-out synthetic observations, not a statistically representative syndicated-loan corpus.
- No proprietary identifiers, legal documents, agent records, or authorized crosswalks were tested.
- Jev probabilities/confidence are model outputs, not demonstrated calibration for real loans.
- Precision here means exact agreement with the synthetic facility policy; it is not proof of production accuracy.
- Production adoption still requires licensed/authorized data, a larger independently labelled set,
  shadow operation, human review, privacy/legal approval, drift monitoring, and false-merge controls.
"""


def run_live(
    *,
    repo_root: Path,
    report_date: str,
    overwrite: bool,
    concurrency: int = 6,
) -> RunSummary:
    reports_dir = repo_root / "reports"
    run_dir_name = f"loan-identity-jev-{report_date}"
    run_dir = reports_dir / run_dir_name
    report_path = reports_dir / f"semantic-loan-mapping-report-{report_date}.md"
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(f"run directory already exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    fixture = build_synthetic_fixture(seed=20260918)
    dataset_bytes = (
        json.dumps(fixture.to_json_dict(), indent=2, sort_keys=True) + "\n"
    ).encode()
    (run_dir / "dataset.json").write_bytes(dataset_bytes)
    dataset_hash = _sha256_bytes(dataset_bytes)

    retriever = CandidateRetriever(TemporalLoanStore(fixture.versions), top_k=4)
    cache_path = run_dir / "live-responses.jsonl"
    cache = ReplayCache(cache_path)
    development = [
        item for item in fixture.observations if item.split == "development"
    ]
    test = [item for item in fixture.observations if item.split == "test"]
    if not os.environ.get("TYPESAFE_API_KEY"):
        raise RuntimeError("TYPESAFE_API_KEY is required for a live run")

    async def execute() -> tuple[list[RawCase], list[RawCase], float]:
        async with AsyncTypeSafeClient(model=MODEL, timeout=15.0) as client:
            evaluator = JevEvaluator(client=client, model=MODEL)
            development_raw, development_wall = await _score_live(
                development,
                retriever,
                evaluator,
                cache,
                concurrency=concurrency,
            )
            test_raw, test_wall = await _score_live(
                test,
                retriever,
                evaluator,
                cache,
                concurrency=concurrency,
            )
        return development_raw, test_raw, development_wall + test_wall

    development_raw, test_raw, live_wall = asyncio.run(execute())
    config, development_metrics = tune_policy(development_raw)
    policy_bytes = (
        json.dumps(config.to_json_dict(), indent=2, sort_keys=True) + "\n"
    ).encode()
    (run_dir / "policy-config.json").write_bytes(policy_bytes)
    policy_hash = _sha256_bytes(policy_bytes)

    development_records = apply_policy(development_raw, config)
    test_records = apply_policy(test_raw, config)
    test_metrics = compute_metrics(test_records)
    acceptance_result = acceptance(test_metrics)
    _write_jsonl(
        run_dir / "live-outcomes.jsonl",
        [record.to_json_dict() for record in (*development_records, *test_records)],
    )

    replay_cache = ReplayCache(cache_path)
    replay_development = _replay_cases(
        development, retriever, replay_cache, model=MODEL
    )
    replay_test = _replay_cases(test, retriever, replay_cache, model=MODEL)
    replay_development_records = apply_policy(replay_development, config)
    replay_test_records = apply_policy(replay_test, config)
    replay_metrics = compute_metrics(replay_test_records)
    _write_jsonl(
        run_dir / "replay-outcomes.jsonl",
        [
            record.to_json_dict()
            for record in (*replay_development_records, *replay_test_records)
        ],
    )
    replay_identical = _quality_projection(test_metrics) == _quality_projection(
        replay_metrics
    )

    metrics_bundle = {
        "development": compute_metrics(development_records),
        "test_live": test_metrics,
        "test_replay": replay_metrics,
        "acceptance": acceptance_result,
        "replay_quality_identical": replay_identical,
        "live_wall_seconds": live_wall,
    }
    _write_json(run_dir / "metrics.json", metrics_bundle)
    git_commit = _git_commit(repo_root)
    live_response_bytes = cache_path.read_bytes()
    manifest = {
        "report_date": report_date,
        "execution_started_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit,
        "python": sys.version,
        "platform": platform.platform(),
        "model_requested": MODEL,
        "models_returned": sorted(
            {
                raw.result.model
                for raw in (*development_raw, *test_raw)
                if raw.result is not None
            }
        ),
        "dataset_sha256": dataset_hash,
        "policy_sha256": policy_hash,
        "live_responses_sha256": _sha256_bytes(live_response_bytes),
        "development_rows": len(development),
        "test_rows": len(test),
        "concurrency": concurrency,
        "synthetic_only": True,
    }
    _write_json(run_dir / "manifest.json", manifest)
    report = _render_report(
        report_date=report_date,
        git_commit=git_commit,
        fixture=fixture,
        dataset_hash=dataset_hash,
        policy=config,
        policy_hash=policy_hash,
        development_metrics=development_metrics,
        test_metrics=test_metrics,
        replay_metrics=replay_metrics,
        acceptance_result=acceptance_result,
        live_wall_seconds=live_wall,
        live_response_hash=manifest["live_responses_sha256"],
        replay_identical=replay_identical,
        run_dir_name=run_dir_name,
    )
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return RunSummary(
        report_path=report_path,
        run_dir=run_dir,
        acceptance_passed=acceptance_result["passed"],
        metrics=metrics_bundle,
    )


def run_replay(*, run_dir: Path) -> dict[str, Any]:
    fixture = SyntheticFixture.from_json_dict(
        json.loads((run_dir / "dataset.json").read_text(encoding="utf-8"))
    )
    config = PolicyConfig(
        **json.loads((run_dir / "policy-config.json").read_text(encoding="utf-8"))
    )
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    cache = ReplayCache(run_dir / "live-responses.jsonl")
    retriever = CandidateRetriever(TemporalLoanStore(fixture.versions), top_k=4)
    test = [item for item in fixture.observations if item.split == "test"]
    raw = _replay_cases(test, retriever, cache, model=manifest["model_requested"])
    metrics = compute_metrics(apply_policy(raw, config))
    expected = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))[
        "test_live"
    ]
    identical = _quality_projection(metrics) == _quality_projection(expected)
    if not identical:
        raise RuntimeError("replay quality metrics differ from the live run")
    return {"quality_identical": True, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run or replay the synthetic Jev loan-identity benchmark."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    live = subparsers.add_parser("live")
    live.add_argument("--repo-root", type=Path, default=Path.cwd())
    live.add_argument("--report-date", required=True)
    live.add_argument("--overwrite", action="store_true")
    replay = subparsers.add_parser("replay")
    replay.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "live":
        summary = run_live(
            repo_root=args.repo_root.resolve(),
            report_date=args.report_date,
            overwrite=args.overwrite,
        )
        print(
            json.dumps(
                {
                    "acceptance_passed": summary.acceptance_passed,
                    "report": str(summary.report_path),
                    "run_dir": str(summary.run_dir),
                },
                sort_keys=True,
            )
        )
    else:
        print(json.dumps(run_replay(run_dir=args.run_dir.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
