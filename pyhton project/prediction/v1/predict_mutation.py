#!/usr/bin/env python3
"""
Inference script for FBN1 Mutation Severity Prediction

This script predicts the pathogenic severity of FBN1 mutations using a pre-trained SVM model.

Usage:
    # Command line
    python predict_mutation.py --position 8606 --base C
    
    # Or programmatic
    from predict_mutation import PredictionPipeline
    pipeline = PredictionPipeline()
    result = pipeline.predict(8606, 'C')
"""

import joblib
import argparse
from pathlib import Path
import sys


class PredictionPipeline:
    """Pipeline for predicting mutation severity"""
    
    def __init__(self, model_dir='.', fasta_file='NM_000138.5.fasta'):
        """
        Initialize prediction pipeline
        
        Args:
            model_dir: Directory containing saved models
            fasta_file: Path to reference FASTA file
        """
        self.model_dir = Path(model_dir)
        
        # Load models
        self.clf = joblib.load(self.model_dir / 'svm_mutation_classifier.pkl')
        self.vectorizer = joblib.load(self.model_dir / 'tfidf_vectorizer.pkl')
        self.encoder = joblib.load(self.model_dir / 'label_encoder.pkl')
        
        # Load reference sequence
        self.reference_seq = self._load_fasta(fasta_file)
    
    def _load_fasta(self, fasta_file):
        """Load sequence from FASTA file"""
        seq = ""
        with open(fasta_file, 'r') as f:
            for line in f:
                if not line.startswith('>'):
                    seq += line.strip()
        return seq.upper()
    
    def _kmerize(self, seq, k=3):
        """Convert sequence to space-separated k-mers"""
        return " ".join([seq[i:i+k] for i in range(len(seq)-k+1)])
    
    def _mutate_sequence(self, position, new_base):
        """
        Create mutated sequence
        
        Args:
            position: 1-indexed position in sequence
            new_base: New nucleotide (A, C, G, or T)
        
        Returns:
            Mutated sequence string
        """
        if position < 1 or position > len(self.reference_seq):
            raise ValueError(f"Position {position} out of range [1, {len(self.reference_seq)}]")
        
        original_base = self.reference_seq[position - 1]
        
        if original_base == new_base:
            raise ValueError(f"Position {position}: reference already has {new_base}")
        
        seq_list = list(self.reference_seq)
        seq_list[position - 1] = new_base
        return ''.join(seq_list)
    
    def predict(self, position, new_base):
        """
        Predict mutation severity
        
        Args:
            position: 1-indexed position in reference sequence
            new_base: New nucleotide (A, C, G, or T)
        
        Returns:
            dict with prediction, confidence, and probabilities
        """
        # Validate inputs
        if new_base not in ['A', 'C', 'G', 'T']:
            raise ValueError(f"Invalid base: {new_base}. Must be A, C, G, or T")
        
        # Generate mutant sequence
        mutant_seq = self._mutate_sequence(position, new_base)
        
        # Convert to k-mers
        kmer_seq = self._kmerize(mutant_seq)
        
        # Vectorize
        X = self.vectorizer.transform([kmer_seq])
        
        # Predict
        pred_idx = self.clf.predict(X)[0]
        pred_label = self.encoder.inverse_transform([pred_idx])[0]
        
        # Get probabilities
        probs = self.clf.predict_proba(X)[0]
        
        original_base = self.reference_seq[position - 1]
        
        return {
            'position': position,
            'original_base': original_base,
            'new_base': new_base,
            'prediction': pred_label,
            'confidence': float(probs[pred_idx]),
            'probabilities': {
                self.encoder.classes_[i]: float(probs[i])
                for i in range(len(self.encoder.classes_))
            }
        }


def format_result(result):
    """Format prediction result for display"""
    output = []
    output.append("=" * 70)
    output.append("MUTATION SEVERITY PREDICTION")
    output.append("=" * 70)
    output.append(
        f"\nMutation: Position {result['position']}, "
        f"{result['original_base']} → {result['new_base']}"
    )
    output.append(f"\n✓ Prediction: {result['prediction']}")
    output.append(f"  Confidence: {result['confidence']:.4f}")
    output.append("\n  Probability breakdown:")
    
    # Sort by probability descending
    sorted_probs = sorted(
        result['probabilities'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    for class_name, prob in sorted_probs:
        bar_length = int(prob * 30)
        bar = '█' * bar_length + '░' * (30 - bar_length)
        output.append(f"    {class_name:30s} {prob:.4f}  {bar}")
    
    output.append("\n" + "=" * 70)
    
    return '\n'.join(output)


def main():
    parser = argparse.ArgumentParser(
        description='Predict pathogenic severity of FBN1 mutations'
    )
    parser.add_argument(
        '--position',
        type=int,
        required=True,
        help='1-indexed position in NM_000138.5 sequence'
    )
    parser.add_argument(
        '--base',
        type=str,
        required=True,
        choices=['A', 'C', 'G', 'T'],
        help='New nucleotide at position'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default='.',
        help='Directory containing trained models'
    )
    parser.add_argument(
        '--fasta',
        type=str,
        default='NM_000138.5.fasta',
        help='Reference FASTA file'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        print("Loading models...", file=sys.stderr)
        pipeline = PredictionPipeline(model_dir=args.model_dir, fasta_file=args.fasta)
        
        # Make prediction
        print("Making prediction...", file=sys.stderr)
        result = pipeline.predict(args.position, args.base)
        
        # Output results
        if args.json:
            import json
            print(json.dumps(result, indent=2))
        else:
            print(format_result(result))
    
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print(f"Make sure model files are in {args.model_dir}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
