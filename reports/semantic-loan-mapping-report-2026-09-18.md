# Jev semantic loan identity benchmark report — 2026-09-18

- **Decision:** PASS
- **Execution date:** 2026-09-18 UTC
- **Benchmark code commit:** `13347e5450a376dd044dad838b8ba947eb1279f6`
**Jev model pinned:** `jev-1.13.0`

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

- Version: `synthetic-bitemporal-loans-v1`
- Seed: `20260918`
- Bitemporal version rows: 180
- Development observations: 30, grouped into 10 borrower families
- Held-out test observations: 60, grouped into 20 disjoint borrower families
- Test positives: 40; hard negatives/no-match: 20
- Dataset SHA-256: `62cfcc60744e0b616e1beee5d1db02f952328a261fcbe0553a74d952d5a06316`
- Identity rule: Same canonical facility across amendments is a match; different facility, tranche, lien, currency, refinancing, sponsor, or affiliate is not a strict match.

Cases include typos, abbreviations, legal-suffix variants, old amendment terms, missing
fields, same-borrower sibling facilities/tranches/liens, sponsor-versus-borrower confusion,
currency conflicts, refinancing, near-name affiliates, OCR-like corruption, and unseen
second-lien facilities.

## Held-out live results

- Retrieval Recall@1: **100.00%**
- Retrieval Recall@4: **100.00%**
- Jev raw top-1 accuracy, including no-match: **100.00%**
- Confirmed precision: **100.00%** (39/39; 95% Wilson interval 91.03%–100.00%)
- Confirmed recall: **97.50%** (39/40)
- Confirmed coverage: **65.00%** (39/60)
- False merges: **0**; false-merge rate **0.00%**
- Review rate: **1.67%**; no-match rate: **33.33%**
- Provider errors: **0**
- Live latency: p50 **1.837s**, p95 **2.796s**, p99 **2.869s**, max **2.869s**
- Live scored wall time: **5.615s** for 90 development + test calls
- Throughput at concurrency 6: **16.03 observations/s**

### Scenario slices

- `amendment_current`: n=4, precision=100.0%, recall=100.0%, matches=4, FP=0
- `amount_drift`: n=4, precision=100.0%, recall=100.0%, matches=4, FP=0
- `currency_conflict_no_match`: n=4, precision=0.0%, recall=0.0%, matches=0, FP=0
- `facility_alias`: n=8, precision=100.0%, recall=100.0%, matches=8, FP=0
- `legal_suffix_variation`: n=4, precision=100.0%, recall=100.0%, matches=4, FP=0
- `missing_fields`: n=4, precision=100.0%, recall=100.0%, matches=4, FP=0
- `near_name_no_match`: n=4, precision=0.0%, recall=0.0%, matches=0, FP=0
- `ocr_noise`: n=4, precision=100.0%, recall=100.0%, matches=4, FP=0
- `refinancing_no_match`: n=4, precision=0.0%, recall=0.0%, matches=0, FP=0
- `same_borrower_sibling`: n=4, precision=100.0%, recall=100.0%, matches=4, FP=0
- `sponsor_confusion_no_match`: n=4, precision=0.0%, recall=0.0%, matches=0, FP=0
- `stale_amendment`: n=4, precision=100.0%, recall=100.0%, matches=4, FP=0
- `typo_alias`: n=4, precision=100.0%, recall=75.0%, matches=3, FP=0
- `unseen_second_lien_no_match`: n=4, precision=0.0%, recall=0.0%, matches=0, FP=0

## Development-only policy selection

The decision thresholds were selected on the 30-row development split before evaluating
the 60-row borrower-disjoint test split. The test split was not used by `tune_policy`.
Development confirmed precision was 100.00% and
confirmed recall was 95.00%.

```json
{
  "max_material_conflict": 0.35,
  "min_choice_margin": 0.0,
  "min_choice_probability": 0.35,
  "min_relationship_confidence": 0.0,
  "min_relationship_score": 1.2,
  "min_same_borrower": 0.35,
  "min_same_facility": 0.35
}
```

Policy SHA-256: `1ae20d0aa0dbfebda9ba06bc5edc6a647915787b1ff5dd28472ea85d342869c9`

## Acceptance gates

- PASS: `confirmed_precision_at_least_90pct`
- PASS: `confirmed_recall_at_least_80pct`
- PASS: `retrieval_recall_at_k_at_least_98pct`
- PASS: `jev_top1_accuracy_at_least_90pct`
- PASS: `false_merge_rate_at_most_10pct`
- PASS: `zero_false_merges`
- PASS: `live_p95_at_most_10_seconds`
- PASS: `zero_provider_errors`

## Replay proof

The saved sanitized response cache was replayed with no API calls. Decisions and all quality
metrics were byte-equivalent after excluding timing fields.
Replay p50/p95 were 0.010392s / 0.012782s.
Live response-cache SHA-256: `4a344d7c28c146988168875ec72fc757c250121bb83090b267b1a69e7784febe`.

## Reproduce

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .

# Offline: tests and exact saved-result replay need no key.
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m loan_identity_jev.benchmark replay \
  --run-dir reports/loan-identity-jev-2026-09-18

# Live rerun: synthetic data only. This spends TypeSafe credits.
export TYPESAFE_API_KEY=...  # never commit it
.venv/bin/python -m loan_identity_jev.benchmark live \
  --repo-root . --report-date 2026-09-18 --overwrite
```

A live rerun can differ because hosted model/service behavior can change. The benchmark pins
`jev-1.13.0` and records response and configuration hashes so drift is visible.

## Artifacts

- `reports/loan-identity-jev-2026-09-18/dataset.json`
- `reports/loan-identity-jev-2026-09-18/policy-config.json`
- `reports/loan-identity-jev-2026-09-18/live-responses.jsonl`
- `reports/loan-identity-jev-2026-09-18/live-outcomes.jsonl`
- `reports/loan-identity-jev-2026-09-18/replay-outcomes.jsonl`
- `reports/loan-identity-jev-2026-09-18/metrics.json`
- `reports/loan-identity-jev-2026-09-18/manifest.json`

## Limits

- This is 60 held-out synthetic observations, not a statistically representative syndicated-loan corpus.
- No proprietary identifiers, legal documents, agent records, or authorized crosswalks were tested.
- Jev probabilities/confidence are model outputs, not demonstrated calibration for real loans.
- Precision here means exact agreement with the synthetic facility policy; it is not proof of production accuracy.
- Production adoption still requires licensed/authorized data, a larger independently labelled set,
  shadow operation, human review, privacy/legal approval, drift monitoring, and false-merge controls.
