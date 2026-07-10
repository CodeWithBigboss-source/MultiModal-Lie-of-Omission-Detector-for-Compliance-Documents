"""
CLI script to run the gold evaluation set.

Usage:
    python scripts/run_eval.py
    python scripts/run_eval.py --domain vehicle_insurance
    python scripts/run_eval.py --max_cases 5
    python scripts/run_eval.py --output outputs/eval_results.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.utils.model_client import TextModelClient, VisionModelClient
from src.eval.evaluator import run_evaluation, format_report


def main():
    parser = argparse.ArgumentParser(description="Run gold eval set")
    parser.add_argument("--domain",    type=str,  default=None)
    parser.add_argument("--max_cases", type=int,  default=None)
    parser.add_argument("--output",    type=str,  default="outputs/eval_results.json")
    args = parser.parse_args()

    text_client   = TextModelClient()
    vision_client = VisionModelClient()

    print("Running evaluation...")
    report = run_evaluation(
        gold_cases_path="data/gold_eval/cases.json",
        images_dir="data/sample_images",
        text_client=text_client,
        vision_client=vision_client,
        max_cases=args.max_cases,
        domain_filter=args.domain,
    )

    # Print to terminal
    print(format_report(report))

    # Save JSON
    os.makedirs("outputs", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(
            {
                "overall_accuracy":          report.overall_accuracy,
                "total_cases":               report.total_cases,
                "total_claims":              report.total_claims,
                "correct_claims":            report.correct_claims,
                "failed_cases":              report.failed_cases,
                "domain_accuracy":           report.domain_accuracy,
                "verdict_precision":         report.verdict_precision,
                "verdict_recall":            report.verdict_recall,
                "avg_confidence_correct":    report.avg_confidence_correct,
                "avg_confidence_incorrect":  report.avg_confidence_incorrect,
                "avg_elapsed_seconds":       report.avg_elapsed_seconds,
                "failure_analysis":          report.failure_analysis,
            },
            f, indent=2
        )
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()