"""
Gold Evaluation Set Runner — Phase 6.

Runs the pipeline against hand-labeled gold cases and measures:
- Per-domain accuracy
- Per-verdict-type precision and recall
- Overall accuracy
- Confidence calibration
- Failure analysis
"""

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from PIL import Image

from src.utils.model_client import TextModelClient, VisionModelClient
from src.utils.schemas import Domain
from src.pipeline import run_pipeline


DOMAIN_MAP = {
    "vehicle_insurance":               Domain.VEHICLE_INSURANCE,
    "health_insurance":                Domain.HEALTH_INSURANCE,
    "loan_application":                Domain.LOAN_APPLICATION,
    "evidence_review":                 Domain.EVIDENCE_REVIEW,
    "licensing_employee_verification": Domain.LICENSING_EMPLOYEE_VERIFICATION,
}


@dataclass
class CaseResult:
    case_id: str
    domain: str
    expected: dict[str, str]
    actual: dict[str, str]
    confidence: dict[str, float]
    correct: int = 0
    total: int = 0
    elapsed: float = 0.0
    error: str = ""


@dataclass
class EvalReport:
    total_cases: int = 0
    total_claims: int = 0
    correct_claims: int = 0
    failed_cases: int = 0
    overall_accuracy: float = 0.0
    domain_accuracy: dict = field(default_factory=dict)
    verdict_precision: dict = field(default_factory=dict)
    verdict_recall: dict = field(default_factory=dict)
    avg_confidence_correct: float = 0.0
    avg_confidence_incorrect: float = 0.0
    avg_elapsed_seconds: float = 0.0
    case_results: list[CaseResult] = field(default_factory=list)
    failure_analysis: list[str] = field(default_factory=list)


def _match_verdict_to_expected(
    claim_text: str,
    expected_verdicts: dict[str, str],
) -> str | None:
    """
    Match a pipeline-produced claim text to an expected verdict key.
    Uses word overlap since claim text may be slightly rephrased.
    """
    claim_words = set(claim_text.lower().split())
    best_match = None
    best_overlap = 0

    for key in expected_verdicts:
        key_words = set(key.lower().split())
        overlap = len(claim_words & key_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_match = key

    # Require at least 2 words overlap to count as a match
    if best_overlap >= 2:
        return best_match
    return None


def run_evaluation(
    gold_cases_path: str,
    images_dir: str,
    text_client: TextModelClient,
    vision_client: VisionModelClient,
    max_cases: int | None = None,
    domain_filter: str | None = None,
) -> EvalReport:
    """
    Run evaluation against gold cases.

    gold_cases_path: path to cases.json
    images_dir: directory containing evidence images
    max_cases: limit number of cases (useful for quick tests)
    domain_filter: only run cases for this domain
    """
    with open(gold_cases_path, "r") as f:
        cases = json.load(f)

    if domain_filter:
        cases = [c for c in cases if c["domain"] == domain_filter]
    if max_cases:
        cases = cases[:max_cases]

    report = EvalReport(total_cases=len(cases))

    # Per-verdict tracking for precision/recall
    verdict_types = [
        "Supported", "Partially Supported", "Contradicted",
        "Insufficient Evidence", "Missing Expected Evidence"
    ]
    tp: dict[str, int] = {v: 0 for v in verdict_types}
    fp: dict[str, int] = {v: 0 for v in verdict_types}
    fn: dict[str, int] = {v: 0 for v in verdict_types}

    domain_correct: dict[str, int] = {}
    domain_total:   dict[str, int] = {}
    elapsed_times:  list[float] = []
    conf_correct:   list[float] = []
    conf_incorrect: list[float] = []

    for case in cases:
        case_id = case["case_id"]
        domain  = DOMAIN_MAP.get(case["domain"])

        if domain is None:
            print(f"Skipping {case_id}: unknown domain {case['domain']}")
            continue

        # Load image
        img_path = os.path.join(images_dir, case["image_filename"])
        if not os.path.exists(img_path):
            print(f"Skipping {case_id}: image not found at {img_path}")
            report.failed_cases += 1
            report.failure_analysis.append(
                f"{case_id}: image file {case['image_filename']} not found in {images_dir}"
            )
            continue

        try:
            image  = Image.open(img_path)
            images = {"eval_img": image}
        except Exception as e:
            print(f"Skipping {case_id}: cannot open image — {e}")
            report.failed_cases += 1
            continue

        # Run pipeline
        start = time.time()
        try:
            pipeline_report = run_pipeline(
                text_client=text_client,
                vision_client=vision_client,
                document_id=case_id,
                domain=domain,
                document_text=case["document_text"],
                images=images,
            )
            elapsed = time.time() - start
            elapsed_times.append(elapsed)
        except Exception as e:
            report.failed_cases += 1
            report.failure_analysis.append(f"{case_id}: pipeline error — {e}")
            continue

        # Compare verdicts
        expected = case["expected_verdicts"]
        actual_map:     dict[str, str]   = {}
        confidence_map: dict[str, float] = {}
        case_correct = 0
        case_total   = 0

        for cv in pipeline_report.claim_verdicts:
            if isinstance(cv, dict):
                claim_text = cv["claim_text"]
                verdict    = cv["verdict"]
                confidence = cv["confidence"]
            else:
                claim_text = cv.claim_text
                verdict    = cv.verdict.value
                confidence = cv.confidence

            matched_key = _match_verdict_to_expected(claim_text, expected)
            if matched_key is None:
                continue

            actual_map[matched_key]     = verdict
            confidence_map[matched_key] = confidence
            expected_verdict            = expected[matched_key]

            case_total += 1
            report.total_claims += 1
            report.correct_claims += (1 if verdict == expected_verdict else 0)

            if verdict == expected_verdict:
                case_correct += 1
                tp[verdict] += 1
                conf_correct.append(confidence)
            else:
                fp[verdict] += 1
                fn[expected_verdict] += 1
                conf_incorrect.append(confidence)
                report.failure_analysis.append(
                    f"{case_id} [{matched_key}]: "
                    f"expected={expected_verdict}, got={verdict} "
                    f"(conf={confidence:.0%})"
                )

            # Domain tracking
            dom_key = case["domain"]
            domain_correct[dom_key] = domain_correct.get(dom_key, 0) + (
                1 if verdict == expected_verdict else 0
            )
            domain_total[dom_key] = domain_total.get(dom_key, 0) + 1

        case_result = CaseResult(
            case_id=case_id,
            domain=case["domain"],
            expected=expected,
            actual=actual_map,
            confidence=confidence_map,
            correct=case_correct,
            total=case_total,
            elapsed=elapsed,
        )
        report.case_results.append(case_result)

    # Compute summary metrics
    if report.total_claims > 0:
        report.overall_accuracy = report.correct_claims / report.total_claims

    for dom in domain_total:
        report.domain_accuracy[dom] = (
            domain_correct.get(dom, 0) / domain_total[dom]
            if domain_total[dom] > 0 else 0.0
        )

    for v in verdict_types:
        precision_denom = tp[v] + fp[v]
        recall_denom    = tp[v] + fn[v]
        report.verdict_precision[v] = (
            tp[v] / precision_denom if precision_denom > 0 else 0.0
        )
        report.verdict_recall[v] = (
            tp[v] / recall_denom if recall_denom > 0 else 0.0
        )

    report.avg_confidence_correct   = (
        sum(conf_correct) / len(conf_correct) if conf_correct else 0.0
    )
    report.avg_confidence_incorrect = (
        sum(conf_incorrect) / len(conf_incorrect) if conf_incorrect else 0.0
    )
    report.avg_elapsed_seconds = (
        sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0.0
    )

    return report


def format_report(report: EvalReport) -> str:
    """Format eval report as readable text for terminal or file."""
    lines = [
        "=" * 60,
        "GOLD EVALUATION REPORT",
        "=" * 60,
        f"Total cases:        {report.total_cases}",
        f"Failed cases:       {report.failed_cases}",
        f"Total claims:       {report.total_claims}",
        f"Correct claims:     {report.correct_claims}",
        f"Overall accuracy:   {report.overall_accuracy:.1%}",
        f"Avg time/case:      {report.avg_elapsed_seconds:.1f}s",
        f"Avg conf (correct): {report.avg_confidence_correct:.1%}",
        f"Avg conf (wrong):   {report.avg_confidence_incorrect:.1%}",
        "",
        "── Domain Accuracy ─────────────────────────────────",
    ]
    for dom, acc in report.domain_accuracy.items():
        lines.append(f"  {dom:<42} {acc:.1%}")

    lines += ["", "── Verdict Precision / Recall ───────────────────────"]
    for v in report.verdict_precision:
        p = report.verdict_precision[v]
        r = report.verdict_recall[v]
        lines.append(f"  {v:<30} P={p:.1%}  R={r:.1%}")

    if report.failure_analysis:
        lines += ["", "── Failure Analysis ─────────────────────────────────"]
        for f in report.failure_analysis:
            lines.append(f"  {f}")

    lines.append("=" * 60)
    return "\n".join(lines)