#!/bin/bash
# Usage: bash batch_cancel.sh <start_job_id> <end_job_id>
# Example: bash batch_cancel.sh 2726932 2726944

START=${1:?Usage: bash batch_cancel.sh <start_job_id> <end_job_id>}
END=${2:?Usage: bash batch_cancel.sh <start_job_id> <end_job_id>}

for i in $(seq $START $END); do
    scancel $i
done

echo "Cancelled jobs $START to $END"
