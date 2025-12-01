#!/bin/bash
# batch_test_mutations.sh
# Test multiple mutations with predict_mutation_v2.py

echo "=========================================="
echo "FBN1 Mutation Prediction Batch Test"
echo "=========================================="
echo ""

# Array of mutations to test
declare -a mutations=(
    "8606:A"
    "8606:C"
    "8606:G"
    "8595:A"
    "8595:C"
    "8595:G"
    "8579:A"
    "8579:G"
    "8567:A"
    "8555:A"
)

# Create output directory
mkdir -p batch_results

echo "Testing ${#mutations[@]} mutations..."
echo ""

# Loop through mutations
for mut in "${mutations[@]}"
do
    IFS=':' read -r position base <<< "$mut"
    echo "Testing Position $position → $base"

    python predict_mutation_v2.py \
        --position $position \
        --base $base \
        --output batch_results \
        --json > batch_results/result_${position}_${base}.json 2>&1

    echo "  ✓ Saved to batch_results/result_${position}_${base}.json"
    echo ""
done

echo "=========================================="
echo "Batch test complete!"
echo "Results saved to batch_results/"
echo "=========================================="
