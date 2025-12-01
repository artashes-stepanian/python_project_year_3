#!/usr/bin/env python3
"""
QUICK START GUIDE - FBN1 Mutation Severity SVM Predictor

This is a complete working example showing how to use the pre-trained model.
Run this script to see predictions for different mutations.
"""

import sys
from predict_mutation import PredictionPipeline, format_result


def main():
    """Run example predictions"""
    
    print("\n" + "=" * 80)
    print("FBN1 MUTATION SEVERITY PREDICTOR - QUICK START")
    print("=" * 80)
    
    # Initialize pipeline (loads models)
    print("\nInitializing prediction pipeline...")
    try:
        pipeline = PredictionPipeline()
    except FileNotFoundError as e:
        print(f"Error: Could not load models - {e}")
        print("Make sure these files are in the current directory:")
        print("  - svm_mutation_classifier.pkl")
        print("  - tfidf_vectorizer.pkl")
        print("  - label_encoder.pkl")
        print("  - NM_000138.5.fasta")
        sys.exit(1)
    
    print("✓ Models loaded successfully")
    print(f"  Reference sequence length: {len(pipeline.reference_seq)} bp")
    print(f"  Trained classes: {', '.join(pipeline.encoder.classes_)}")
    
    # Example mutations
    mutations = [
        (8606, 'C', 'c.8606T>C - L2869S (from ClinVar)'),
        (8595, 'C', 'c.8595A>C - K2865N (from ClinVar)'),
        (8579, 'G', 'c.8579A>G - D2860G (from ClinVar)'),
        (5000, 'G', 'Random test mutation'),
        (1000, 'A', 'Another random mutation'),
    ]
    
    print("\n" + "=" * 80)
    print("EXAMPLE PREDICTIONS")
    print("=" * 80)
    
    for position, base, description in mutations:
        try:
            print(f"\n[{description}]")
            result = pipeline.predict(position, base)
            
            # Print formatted output
            print(f"  Position: {position}")
            print(f"  Change: {result['original_base']} → {result['new_base']}")
            print(f"  Prediction: {result['prediction']}")
            print(f"  Confidence: {result['confidence']:.4f}")
            
            # Show top 3 predictions
            sorted_probs = sorted(
                result['probabilities'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            print(f"  Top predictions:")
            for i, (label, prob) in enumerate(sorted_probs[:3], 1):
                print(f"    {i}. {label}: {prob:.4f}")
        
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n" + "=" * 80)
    print("USING THE PREDICTOR IN YOUR CODE")
    print("=" * 80)
    
    print("""
# Example 1: Simple prediction
from predict_mutation import PredictionPipeline

pipeline = PredictionPipeline()
result = pipeline.predict(position=8606, new_base='C')

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.4f}")

# Example 2: Batch predictions
mutations_to_predict = [
    (8606, 'C'),
    (8595, 'C'),
    (1000, 'A'),
]

results = []
for pos, base in mutations_to_predict:
    result = pipeline.predict(pos, base)
    results.append(result)
    print(f"Position {pos}: {result['prediction']}")

# Example 3: Access all probabilities
result = pipeline.predict(8606, 'C')
for class_name, probability in result['probabilities'].items():
    print(f"{class_name}: {probability:.4f}")

# Example 4: Command line usage
# python predict_mutation.py --position 8606 --base C
# python predict_mutation.py --position 8606 --base C --json

# Example 5: Get JSON output for integration
import json
result = pipeline.predict(8606, 'C')
json_output = json.dumps(result, indent=2)
print(json_output)
    """)
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("""
1. To train your own model with real data:
   python train_svm_model.py --input your_data.csv --output_dir ./my_models

2. To get predictions for specific mutations:
   python predict_mutation.py --position <pos> --base <A|C|G|T>

3. To integrate into your pipeline:
   from predict_mutation import PredictionPipeline
   pipeline = PredictionPipeline()
   result = pipeline.predict(<position>, '<base>')

4. For batch processing:
   python predict_mutation.py --position 8606 --base C > results.txt
   
5. For JSON output (API integration):
   python predict_mutation.py --position 8606 --base C --json

6. Read the README.md for detailed documentation
    """)
    
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
