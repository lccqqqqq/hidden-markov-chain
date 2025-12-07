#!/bin/bash
# Script to concatenate rank files for all mess3 datasets

set -e  # Exit on error

echo "=========================================="
echo "Concatenating mess3 datasets"
echo "=========================================="
echo ""

# Loop through all mess3 dataset directories
for dataset_dir in data/datasets/mess3*/; do
    echo "Processing dataset: $dataset_dir"

    # Process train, val, and test splits
    for split in train val test; do
        split_dir="${dataset_dir}${split}"

        if [ -d "$split_dir" ]; then
            echo "  Processing $split split..."

            # Check if rank files exist
            if ls "$split_dir"/observations_rank*.pt 1> /dev/null 2>&1; then
                python concatenate_ranks.py "$split_dir"
            else
                echo "    No rank files found in $split_dir (may already be concatenated)"
            fi
        else
            echo "    $split_dir does not exist, skipping"
        fi
    done

    echo ""
done

echo "=========================================="
echo "All datasets processed"
echo "=========================================="
