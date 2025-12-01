#!/usr/bin/env python3
"""
Training script for FBN1 Mutation Severity SVM Classifier

This script trains an SVM model on DNA sequences to predict mutation severity.
It can use real training data or generate synthetic data for demonstration.

Usage:
    python train_svm_model.py --input library_head.csv --output_dir ./models
"""

import pandas as pd
import numpy as np
import argparse
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')


def load_reference_sequence(fasta_file):
    """Load reference sequence from FASTA file"""
    seq = ""
    with open(fasta_file, 'r') as f:
        for line in f:
            if not line.startswith('>'):
                seq += line.strip()
    return seq.upper()


def kmerize(seq, k=3):
    """Convert sequence to space-separated k-mers"""
    return " ".join([seq[i:i+k] for i in range(len(seq)-k+1)])


def generate_synthetic_data(reference_seq, num_samples_per_class=30):
    """Generate synthetic training data for demonstration"""
    np.random.seed(42)
    
    severity_classes = [
        'Benign',
        'Likely benign',
        'Uncertain significance',
        'Likely pathogenic',
        'Pathogenic'
    ]
    
    def mutate_sequence(seq, num_mutations=1):
        """Introduce random single nucleotide mutations"""
        seq_list = list(seq)
        bases = ['A', 'C', 'G', 'T']
        
        positions = np.random.choice(len(seq), size=num_mutations, replace=False)
        for pos in positions:
            current_base = seq_list[pos]
            possible_bases = [b for b in bases if b != current_base]
            seq_list[pos] = np.random.choice(possible_bases)
        
        return ''.join(seq_list)
    
    synthetic_data = []
    for severity in severity_classes:
        for i in range(num_samples_per_class):
            # Vary number of mutations based on severity
            if severity == 'Benign':
                num_muts = np.random.randint(1, 3)
            elif severity == 'Likely benign':
                num_muts = np.random.randint(1, 4)
            elif severity == 'Uncertain significance':
                num_muts = np.random.randint(2, 5)
            elif severity == 'Likely pathogenic':
                num_muts = np.random.randint(3, 6)
            else:  # Pathogenic
                num_muts = np.random.randint(4, 8)
            
            mutated_seq = mutate_sequence(reference_seq, num_muts)
            synthetic_data.append({
                'mutant_sequence': mutated_seq,
                'Germline classification': severity
            })
    
    return pd.DataFrame(synthetic_data)


def train_model(df_train, output_dir='./models'):
    """Train SVM classifier on mutation data"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("TRAINING SVM MUTATION SEVERITY CLASSIFIER")
    print("=" * 70)
    
    # Data preparation
    print(f"\nDataset size: {len(df_train)} samples")
    print(f"Class distribution:")
    print(df_train['Germline classification'].value_counts())
    
    # Feature engineering with k-mers
    print("\nApplying k-mer feature extraction (k=3)...")
    df_train['kmer_seq'] = df_train['mutant_sequence'].apply(
        lambda s: kmerize(s.upper(), k=3)
    )
    
    # TF-IDF vectorization
    tfidf_vectorizer = TfidfVectorizer(
        analyzer='word',
        max_features=300,
        ngram_range=(1, 2)
    )
    X = tfidf_vectorizer.fit_transform(df_train['kmer_seq'])
    print(f"Feature matrix shape: {X.shape}")
    
    # Label encoding
    le = LabelEncoder()
    y = le.fit_transform(df_train['Germline classification'])
    
    print(f"\nLabel mapping:")
    for idx, label in enumerate(le.classes_):
        print(f"  {idx}: {label}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train SVM
    print("\nTraining SVM classifier...")
    clf = SVC(
        kernel='rbf',
        C=10.0,
        gamma='scale',
        class_weight='balanced',
        probability=True,
        random_state=42
    )
    clf.fit(X_train, y_train)
    
    # Evaluate
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    
    print(f"\nTraining accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")
    
    # Classification report
    y_pred = clf.predict(X_test)
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Cross-validation
    cv_scores = cross_val_score(clf, X, y, cv=5)
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Save models
    print(f"\nSaving models to {output_dir}...")
    joblib.dump(clf, output_dir / 'svm_mutation_classifier.pkl')
    joblib.dump(tfidf_vectorizer, output_dir / 'tfidf_vectorizer.pkl')
    joblib.dump(le, output_dir / 'label_encoder.pkl')
    print("✓ Models saved successfully")
    
    return clf, tfidf_vectorizer, le


def main():
    parser = argparse.ArgumentParser(
        description='Train SVM classifier for FBN1 mutation severity prediction'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='library_head.csv',
        help='Input CSV file with mutant_sequence and Germline classification columns'
    )
    parser.add_argument(
        '--reference',
        type=str,
        default='NM_000138.5.fasta',
        help='Reference FASTA file (used for synthetic data generation if input has too few samples)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./models',
        help='Directory to save trained models'
    )
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Force generation of synthetic training data'
    )
    
    args = parser.parse_args()
    
    # Load or generate training data
    if args.synthetic or not Path(args.input).exists():
        print(f"Generating synthetic training data from {args.reference}...")
        reference_seq = load_reference_sequence(args.reference)
        df_train = generate_synthetic_data(reference_seq, num_samples_per_class=30)
    else:
        print(f"Loading training data from {args.input}...")
        df_train = pd.read_csv(args.input)
        df_train = df_train.dropna(subset=['mutant_sequence', 'Germline classification'])
        
        # If too few samples, augment with synthetic data
        if len(df_train) < 50:
            print(f"Warning: Only {len(df_train)} samples. Generating synthetic data...")
            reference_seq = load_reference_sequence(args.reference)
            df_synthetic = generate_synthetic_data(reference_seq, num_samples_per_class=20)
            df_train = pd.concat([df_train, df_synthetic], ignore_index=True)
    
    # Train model
    clf, vectorizer, encoder = train_model(df_train, output_dir=args.output_dir)
    
    print("\n" + "=" * 70)
    print("Training complete! Models ready for inference.")
    print("=" * 70)


if __name__ == '__main__':
    main()
