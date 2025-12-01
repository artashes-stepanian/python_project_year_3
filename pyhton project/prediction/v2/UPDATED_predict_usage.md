#!/usr/bin/env python3
"""
UPDATED predict_mutation.py - NEW COMMAND-LINE INTERFACE
=========================================================

This updated version includes flexible path arguments for:
- Model directory location
- FASTA file location  
- Output directory for results

NEW FEATURES
════════════════════════════════════════════════════════════════════════════════

✓ Custom model directory:     --model /path/to/models
✓ Custom FASTA file:          --fasta /path/to/sequence.fasta
✓ Output directory:           --output /path/to/results
✓ JSON export:                --json flag
✓ Automatic result saving:    Timestamped JSON files
✓ Better error messages:      Detailed path validation
✓ Help documentation:         --help shows all options


USAGE EXAMPLES
════════════════════════════════════════════════════════════════════════════════

1. BASIC USAGE (Current directory)
   ────────────────────────────────────────────────────────────────────────
   $ python predict_mutation.py --position 8606 --base C
   
   Prerequisites:
   - All .pkl files in current directory
   - NM_000138.5.fasta in current directory

2. CUSTOM MODEL DIRECTORY
   ────────────────────────────────────────────────────────────────────────
   $ python predict_mutation.py \
       --model /path/to/models \
       --position 8606 \
       --base C
   
   Model directory should contain:
   - svm_mutation_classifier.pkl
   - tfidf_vectorizer.pkl
   - label_encoder.pkl

3. CUSTOM FASTA FILE
   ────────────────────────────────────────────────────────────────────────
   $ python predict_mutation.py \
       --fasta /path/to/NM_000138.5.fasta \
       --position 8606 \
       --base C

4. SAVE RESULTS TO OUTPUT DIRECTORY
   ────────────────────────────────────────────────────────────────────────
   $ python predict_mutation.py \
       --position 8606 \
       --base C \
       --output /path/to/results
   
   Creates timestamped JSON file:
   - results/prediction_8606_C_20251201_112345.json

5. ALL CUSTOM PATHS + JSON OUTPUT
   ────────────────────────────────────────────────────────────────────────
   $ python predict_mutation.py \
       --model /ml/models \
       --fasta /data/sequences/NM_000138.5.fasta \
       --output /results/fbn1 \
       --position 8606 \
       --base C \
       --json

6. GET HELP
   ────────────────────────────────────────────────────────────────────────
   $ python predict_mutation.py --help

7. BATCH PROCESSING WITH CUSTOM PATHS
   ────────────────────────────────────────────────────────────────────────
   #!/bin/bash
   
   MODELS=/ml/fbn1_models
   FASTA=/data/fbn1.fasta
   OUTPUT=/results/predictions
   
   for pos in 8606 8595 8579 1000; do
     for base in A C G T; do
       python predict_mutation.py \
         --model $MODELS \
         --fasta $FASTA \
         --output $OUTPUT \
         --position $pos \
         --base $base \
         --json
     done
   done


COMMAND-LINE ARGUMENTS
════════════════════════════════════════════════════════════════════════════════

REQUIRED ARGUMENTS:
────────────────────────────────────────────────────────────────────────────

  --position POSITION
    1-indexed position in reference sequence
    Example: --position 8606
    Type: integer, range [1, 11609]

  --base BASE
    New nucleotide at position
    Example: --base C
    Type: one of [A, C, G, T]


OPTIONAL ARGUMENTS:
────────────────────────────────────────────────────────────────────────────

  --model MODEL_DIR
    Directory containing trained models
    Default: . (current directory)
    Example: --model /ml/models
    
    Must contain:
    - svm_mutation_classifier.pkl
    - tfidf_vectorizer.pkl
    - label_encoder.pkl

  --fasta FASTA_PATH
    Path to reference FASTA file
    Default: NM_000138.5.fasta (current directory)
    Example: --fasta /data/sequences/NM_000138.5.fasta
    Example: --fasta ./reference/fbn1.fasta

  --output OUTPUT_DIR
    Directory to save results (JSON files)
    Default: None (results only printed to stdout)
    Example: --output /results
    Example: --output ./predictions
    
    Note: Creates directory if it doesn't exist
    Creates timestamped files: prediction_POS_BASE_TIMESTAMP.json

  --json
    Output results as JSON (in addition to formatted text)
    Default: False (formatted text output)
    Example: --json
    
    When used with --output, saves JSON to file
    When used alone, prints JSON to stdout


OUTPUT FORMATS
════════════════════════════════════════════════════════════════════════════════

DEFAULT OUTPUT (Formatted Text to stdout):
────────────────────────────────────────────────────────────────────────────

  ======================================================================
  MUTATION SEVERITY PREDICTION
  ======================================================================

  Mutation: Position 8606, T → C

  ✓ Prediction: Benign
    Confidence: 0.1578

    Probability breakdown:
      Uncertain significance        0.2532  ████████░░░░░░░░░░░░░░░░░░░░
      Pathogenic                    0.2200  ██████░░░░░░░░░░░░░░░░░░░░░░░
      Likely pathogenic             0.1946  █████░░░░░░░░░░░░░░░░░░░░░░░░
      Likely benign                 0.1743  █████░░░░░░░░░░░░░░░░░░░░░░░░
      Benign                        0.1578  ████░░░░░░░░░░░░░░░░░░░░░░░░░

  ======================================================================


JSON OUTPUT (with --json flag):
────────────────────────────────────────────────────────────────────────────

  {
    "position": 8606,
    "original_base": "T",
    "new_base": "C",
    "prediction": "Benign",
    "confidence": 0.1578,
    "probabilities": {
      "Benign": 0.1578,
      "Likely benign": 0.1743,
      "Uncertain significance": 0.2532,
      "Likely pathogenic": 0.1946,
      "Pathogenic": 0.22
    }
  }


SAVED FILE OUTPUT (with --output flag):
────────────────────────────────────────────────────────────────────────────

  File: prediction_8606_C_20251201_112345.json
  
  Location: {output_dir}/prediction_{position}_{base}_{timestamp}.json
  
  Content: Same as JSON output above


ERROR MESSAGES & TROUBLESHOOTING
════════════════════════════════════════════════════════════════════════════════

ERROR: Model directory not found
────────────────────────────────────────────────────────────────────────────
Message: "Model directory not found: /path/to/models"
Solution:
  1. Check path is correct: ls /path/to/models
  2. Verify .pkl files exist:
     ls /path/to/models/*.pkl
  3. Use correct path in --model argument

ERROR: Model file not found
────────────────────────────────────────────────────────────────────────────
Message: "Model file not found: /path/to/models/svm_mutation_classifier.pkl"
Solution:
  1. Check all 3 files exist:
     - svm_mutation_classifier.pkl
     - tfidf_vectorizer.pkl
     - label_encoder.pkl
  2. Verify file permissions: chmod 644 *.pkl
  3. Re-download models if corrupted

ERROR: FASTA file not found
────────────────────────────────────────────────────────────────────────────
Message: "FASTA file not found: /path/to/NM_000138.5.fasta"
Solution:
  1. Check file path: ls /path/to/NM_000138.5.fasta
  2. Verify file exists and is readable
  3. Use correct path in --fasta argument

ERROR: Position out of range
────────────────────────────────────────────────────────────────────────────
Message: "Position 12000 out of range [1, 11609]"
Solution:
  1. Check position is between 1 and 11,609
  2. Position should be 1-indexed (not 0-indexed)
  3. Re-verify mutation location

ERROR: Reference already has [BASE]
────────────────────────────────────────────────────────────────────────────
Message: "Position 8606: reference already has C"
Solution:
  1. New base must differ from reference
  2. At position 8606, reference is 'T', not 'C'
  3. Try: python predict_mutation.py --position 8606 --base A


PYTHON API USAGE
════════════════════════════════════════════════════════════════════════════════

Import the class:
────────────────────────────────────────────────────────────────────────────
  from predict_mutation import PredictionPipeline

Initialize with custom paths:
────────────────────────────────────────────────────────────────────────────
  pipeline = PredictionPipeline(
      model_dir='/ml/models',
      fasta_file='/data/NM_000138.5.fasta'
  )

Make a prediction:
────────────────────────────────────────────────────────────────────────────
  result = pipeline.predict(position=8606, new_base='C')
  
  print(result['prediction'])         # 'Benign'
  print(result['confidence'])         # 0.1578
  print(result['probabilities'])      # Dict of all probabilities

Batch predictions:
────────────────────────────────────────────────────────────────────────────
  mutations = [(8606, 'C'), (8595, 'C'), (1000, 'A')]
  
  results = []
  for pos, base in mutations:
      try:
          result = pipeline.predict(pos, base)
          results.append(result)
      except ValueError as e:
          print(f"Position {pos}: {e}")
  
  # Save to CSV
  import pandas as pd
  df = pd.DataFrame(results)
  df.to_csv('predictions.csv', index=False)

Handle errors:
────────────────────────────────────────────────────────────────────────────
  from predict_mutation import PredictionPipeline
  
  try:
      pipeline = PredictionPipeline(
          model_dir='/path/to/models',
          fasta_file='/path/to/fasta.fa'
      )
  except FileNotFoundError as e:
      print(f"Error: {e}")
      exit(1)
  
  try:
      result = pipeline.predict(8606, 'C')
  except ValueError as e:
      print(f"Invalid input: {e}")


WORKFLOW EXAMPLES
════════════════════════════════════════════════════════════════════════════════

EXAMPLE 1: Single Prediction with All Custom Paths
────────────────────────────────────────────────────────────────────────────
$ mkdir -p ~/fbn1_analysis/results

$ python predict_mutation.py \
    --model ~/fbn1_analysis/models \
    --fasta ~/fbn1_analysis/data/NM_000138.5.fasta \
    --output ~/fbn1_analysis/results \
    --position 8606 \
    --base C \
    --json

Output: 
  ~/fbn1_analysis/results/prediction_8606_C_20251201_112345.json


EXAMPLE 2: Batch Predictions with Different Bases
────────────────────────────────────────────────────────────────────────────
#!/bin/bash

POSITION=8606
MODEL=/ml/models
FASTA=/data/fbn1.fasta
OUTPUT=/results/batch1

mkdir -p $OUTPUT

for BASE in A C G T; do
  python predict_mutation.py \
    --model $MODEL \
    --fasta $FASTA \
    --output $OUTPUT \
    --position $POSITION \
    --base $BASE \
    --json
done

Result: 4 prediction files with different bases


EXAMPLE 3: Scan Multiple Positions
────────────────────────────────────────────────────────────────────────────
#!/bin/bash

MODEL=/ml/models
FASTA=/data/fbn1.fasta
OUTPUT=/results/scan

mkdir -p $OUTPUT

# Scan positions 8600-8610, all bases
for POS in $(seq 8600 8610); do
  for BASE in A C G T; do
    python predict_mutation.py \
      --model $MODEL \
      --fasta $FASTA \
      --output $OUTPUT \
      --position $POS \
      --base $BASE \
      --json 2>/dev/null
  done
done

echo "Scan complete. Results in $OUTPUT"


VERSION INFORMATION
════════════════════════════════════════════════════════════════════════════════

Script Version: 2.0 (Updated with flexible path arguments)
Date: December 1, 2025
Changes from v1.0:
  ✓ Added --model argument for custom model directory
  ✓ Added --fasta argument for custom FASTA file path
  ✓ Added --output argument for result directory
  ✓ Added timestamp to output files
  ✓ Improved error messages with detailed paths
  ✓ Better help documentation with examples

Backward Compatibility:
  ✓ Still works with default paths (current directory)
  ✓ All new arguments are optional


REQUIREMENTS
════════════════════════════════════════════════════════════════════════════════

Python: 3.7+
Libraries:
  - joblib (for model loading)
  - scikit-learn (for SVM)
  - numpy & pandas (dependencies)

Install: pip install joblib scikit-learn numpy pandas


TIPS & BEST PRACTICES
════════════════════════════════════════════════════════════════════════════════

1. Use absolute paths for robustness:
   $ python predict_mutation.py \
       --model /ml/trained_models/fbn1_svm \
       --fasta /data/reference/NM_000138.5.fasta \
       --position 8606 --base C

2. Create output directory before running:
   $ mkdir -p /results/my_predictions
   
3. Save both formatted and JSON output:
   $ python predict_mutation.py ... --output /results --json

4. Log your runs for reproducibility:
   $ python predict_mutation.py ... 2>&1 | tee prediction.log

5. Use error checking in scripts:
   if [ $? -ne 0 ]; then
     echo "Prediction failed!"
     exit 1
   fi

6. Process in parallel for large batches:
   $ parallel python predict_mutation.py --model $MODEL ... \
       ::: {1..11609}


NEXT STEPS
════════════════════════════════════════════════════════════════════════════════

1. Test the new command-line interface:
   python predict_mutation.py --help

2. Make a test prediction:
   python predict_mutation.py --position 8606 --base C

3. Try with custom paths:
   python predict_mutation.py --model ./models --fasta ./NM_000138.5.fasta \
       --position 8606 --base C

4. Save results to file:
   python predict_mutation.py --model ./models --output ./results \
       --position 8606 --base C --json

5. Integrate into your workflow with batch predictions
"""
