#!/usr/bin/env python3
"""
predict_mutation_v2.py - MODIFIED FOR LOCAL CONTEXT (FIXED)

FBN1 Mutation Severity Predictor using SVM with LOCAL CONTEXT FEATURES

KEY MODIFICATION: Uses ±50 bp window around mutation for k-mer extraction
instead of full 11,609 bp sequence. This dramatically improves signal-to-noise.

FIX: Properly handles multi-class SVM decision function output
"""

import argparse
import joblib
import numpy as np
from pathlib import Path
import sys
import json
from datetime import datetime

class PredictionPipeline:
    def __init__(self, model_dir=".", fasta_file="NM_000138.5.fasta"):
        """Initialize prediction pipeline."""
        self.model_dir = Path(model_dir)
        self.fasta_file = Path(fasta_file)

        # Load models
        self.svm_model = joblib.load(self.model_dir / "svm_mutation_classifier.pkl")
        self.vectorizer = joblib.load(self.model_dir / "tfidf_vectorizer.pkl")
        self.label_encoder = joblib.load(self.model_dir / "label_encoder.pkl")

        # Load reference sequence
        with open(self.fasta_file, 'r') as f:
            self.ref_seq = ""
            for line in f:
                if not line.startswith('>'):
                    self.ref_seq += line.strip()
        self.ref_seq = self.ref_seq.upper()
        print(f"✓ Loaded reference sequence: {len(self.ref_seq)} bp")

    def _kmerize(self, seq, k=3, position=None, window=50):
        """
        Convert sequence to k-mers.

        MODIFICATION: If position provided, uses ±window bp around mutation
        instead of full sequence.

        Args:
            seq: DNA sequence
            k: k-mer size (default: 3)
            position: 0-indexed position of mutation (optional)
            window: bp to extract on each side (default: 50)

        Returns:
            Space-separated k-mers string
        """
        if position is not None:
            # Extract local window around mutation
            start = max(0, position - window)
            end = min(len(seq), position + window)
            seq = seq[start:end]
            region_bp = end - start
            print(f"  └─ Using local context: {region_bp} bp window (±{window} bp)")
        else:
            print(f"  └─ Using full sequence: {len(seq)} bp")

        # Extract k-mers
        kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
        return " ".join(kmers)

    def predict(self, position, base):
        """
        Predict mutation severity.

        Args:
            position: 1-indexed position (NM_000138.5 format)
            base: New nucleotide (A, C, G, T)

        Returns:
            Dictionary with prediction results
        """
        # Validate position
        if position < 1 or position > len(self.ref_seq):
            raise ValueError(f"Position {position} out of range [1, {len(self.ref_seq)}]")

        # Convert to 0-indexed
        pos_0 = position - 1
        ref_base = self.ref_seq[pos_0]

        # Validate base change
        if base.upper() == ref_base:
            raise ValueError(f"No change at position {position}: {ref_base}→{base}")

        # Create mutated sequence
        seq_list = list(self.ref_seq)
        seq_list[pos_0] = base.upper()
        mutant_seq = ''.join(seq_list)

        print(f"Position: {position} ({ref_base}→{base.upper()})")

        # Extract k-mers WITH LOCAL CONTEXT (key modification!)
        kmer_seq = self._kmerize(mutant_seq, k=3, position=pos_0, window=50)

        # Vectorize
        print(f"  └─ Vectorizing k-mers...")
        vector = self.vectorizer.transform([kmer_seq])

        # Predict
        print(f"  └─ Running SVM classifier...")
        pred_class = self.svm_model.predict(vector)[0]

        # Get decision scores (FIXED: handles multi-class properly)
        decision_scores = self.svm_model.decision_function(vector)[0]

        # For multi-class SVM, decision_scores is a 1D array of shape (n_classes,)
        # Convert to confidence using softmax-like approach
        try:
            exp_scores = np.exp(decision_scores - np.max(decision_scores))  # Numerical stability
            confidence = float(exp_scores[pred_class] / exp_scores.sum())
        except (IndexError, TypeError):
            # Fallback if decision_scores is scalar
            confidence = float(1.0 / (1.0 + np.exp(-float(decision_scores))))

        # Decode class
        class_name = self.label_encoder.inverse_transform([pred_class])[0]

        # Get all class probabilities
        probs = {}
        all_classes = self.label_encoder.classes_

        try:
            # Use exponential normalization for all classes
            if isinstance(decision_scores, np.ndarray) and len(decision_scores) == len(all_classes):
                exp_scores = np.exp(decision_scores - np.max(decision_scores))
                normalized = exp_scores / exp_scores.sum()
                for i, cls in enumerate(all_classes):
                    probs[cls] = float(normalized[i])
            else:
                # Fallback: equal distribution minus 1 for top class
                for i, cls in enumerate(all_classes):
                    if i == pred_class:
                        probs[cls] = float(confidence)
                    else:
                        probs[cls] = float((1.0 - confidence) / (len(all_classes) - 1))
        except Exception as e:
            # Final fallback
            for i, cls in enumerate(all_classes):
                if i == pred_class:
                    probs[cls] = float(confidence)
                else:
                    probs[cls] = float((1.0 - confidence) / (len(all_classes) - 1))

        result = {
            "mutation": f"{ref_base}{position}{base.upper()}",
            "position": position,
            "reference_base": ref_base,
            "mutant_base": base.upper(),
            "prediction": class_name,
            "confidence": float(confidence),
            "probabilities": probs,
            "timestamp": datetime.now().isoformat(),
            "features": {
                "method": "SVM with k-mer features (3-mers)",
                "context": "±50 bp local window",
                "total_kmers": len(kmer_seq.split()),
                "feature_dimension": vector.shape[1]
            }
        }

        return result

def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="FBN1 Mutation Severity Predictor (v2.0 - Local Context)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  Basic usage (current directory):
    python predict_mutation_v2.py --position 8606 --base C

  Custom paths:
    python predict_mutation_v2.py \
      --model /path/to/models \
      --fasta /path/to/NM_000138.5.fasta \
      --output /path/to/results \
      --position 8606 --base C

  With JSON output:
    python predict_mutation_v2.py \
      --position 8606 --base C \
      --output /results \
      --json
        """
    )

    parser.add_argument("--position", type=int, required=True,
                       help="1-indexed mutation position (1-11609)")
    parser.add_argument("--base", type=str, required=True,
                       help="New nucleotide: A, C, G, or T")
    parser.add_argument("--model", type=str, default=".",
                       help="Directory containing model files (default: current)")
    parser.add_argument("--fasta", type=str, default="NM_000138.5.fasta",
                       help="Path to reference FASTA file")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory for JSON results (optional)")
    parser.add_argument("--json", action="store_true",
                       help="Output as JSON")

    args = parser.parse_args()

    try:
        # Initialize pipeline
        pipeline = PredictionPipeline(model_dir=args.model, fasta_file=args.fasta)

        # Make prediction
        print(f"\n{'='*80}")
        print(f"FBN1 Mutation Severity Prediction (v2.0 - Local Context)")
        print(f"{'='*80}\n")

        result = pipeline.predict(args.position, args.base)

        # Output results
        if args.json:
            json_output = json.dumps(result, indent=2)
            print(f"\n{json_output}")

            # Save to file if output directory specified
            if args.output:
                output_dir = Path(args.output)
                output_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = output_dir / f"prediction_{result['position']}_{result['mutant_base']}_{timestamp}.json"
                with open(filename, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\n✓ Saved to: {filename}")
        else:
            # Formatted text output
            print(f"Mutation: {result['mutation']}")
            print(f"Severity: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"\nProbabilities:")
            for cls, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
                bar = "█" * int(prob * 40)
                print(f"  {cls:20s} {bar:40s} {prob:.1%}")
            print(f"\nFeatures: {result['features']['context']}")

        print(f"{'='*80}\n")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
