#!/usr/bin/env python3
"""
compare_versions.py - Compare v1.0 (full sequence) vs v2.0 (local context)
"""

import json
import argparse
from pathlib import Path

def compare_results(v1_file, v2_file):
    """Compare prediction results from v1.0 and v2.0"""

    with open(v1_file) as f:
        v1_result = json.load(f)

    with open(v2_file) as f:
        v2_result = json.load(f)

    print(f"\n{'='*80}")
    print(f"Prediction Comparison: v1.0 (Full Sequence) vs v2.0 (Local Context)")
    print(f"{'='*80}")
    print(f"Mutation: {v1_result.get('mutation', 'N/A')}")
    print()

    # Compare predictions
    print(f"{'Metric':<25} {'v1.0 (Full Seq)':<25} {'v2.0 (Local ±50bp)':<25}")
    print(f"{'-'*75}")

    print(f"{'Prediction':<25} {v1_result.get('prediction', 'N/A'):<25} {v2_result.get('prediction', 'N/A'):<25}")
    print(f"{'Confidence':<25} {v1_result.get('confidence', 0):.4f} {v2_result.get('confidence', 0):.4f}")

    # Compare probability distributions
    print(f"\n{'Probability Distribution:'}")
    print(f"{'-'*75}")
    print(f"{'Class':<30} {'v1.0':<20} {'v2.0':<20}")
    print(f"{'-'*75}")

    v1_probs = v1_result.get('probabilities', {})
    v2_probs = v2_result.get('probabilities', {})

    for cls in sorted(v1_probs.keys()):
        v1_p = v1_probs.get(cls, 0)
        v2_p = v2_probs.get(cls, 0)
        diff = v2_p - v1_p
        change = "↑" if diff > 0.01 else "↓" if diff < -0.01 else "→"

        print(f"{cls:<30} {v1_p:>6.2%}          {v2_p:>6.2%}          {change}")

    print(f"\n{'='*80}")
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare v1.0 vs v2.0 predictions")
    parser.add_argument("--v1", required=True, help="Path to v1.0 JSON result")
    parser.add_argument("--v2", required=True, help="Path to v2.0 JSON result")

    args = parser.parse_args()

    compare_results(args.v1, args.v2)
