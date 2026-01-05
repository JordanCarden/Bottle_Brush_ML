#!/bin/bash
# Generate extreme test set: 8 property combinations at backbone length 15
# Uses GIN model with equal weights (default 1.0 for all properties)

set -e  # Exit on error

METHOD="gin"
LENGTHS=(15)

# Define the 8 property combinations (all corners of 3D property space)
declare -a COMBOS=(
    "max max max"  # 1. Max Area, Max RG, Max RDF
    "max max min"  # 2. Max Area, Max RG, Min RDF
    "max min max"  # 3. Max Area, Min RG, Max RDF
    "max min min"  # 4. Max Area, Min RG, Min RDF
    "min max max"  # 5. Min Area, Max RG, Max RDF
    "min max min"  # 6. Min Area, Max RG, Min RDF
    "min min max"  # 7. Min Area, Min RG, Max RDF
    "min min min"  # 8. Min Area, Min RG, Min RDF
)

echo "=========================================="
echo "Running 8 Extreme Optimization Runs"
echo "Method: GIN"
echo "Backbone Length: 15"
echo "Property Combinations: All 8 corners"
echo "=========================================="
echo

# Create log directory if it doesn't exist
mkdir -p log

RUN_NUM=0

for LENGTH in "${LENGTHS[@]}"; do
    for COMBO in "${COMBOS[@]}"; do
        RUN_NUM=$((RUN_NUM + 1))

        # Parse combo into individual modes
        read -r AREA_MODE RG_MODE RDF_MODE <<< "$COMBO"

        echo "----------------------------------------"
        echo "Run $RUN_NUM/8: Length=$LENGTH, Area=$AREA_MODE, RG=$RG_MODE, RDF=$RDF_MODE"
        echo "----------------------------------------"

        python3 optimization.py \
            --method "$METHOD" \
            --length "$LENGTH" \
            --use-area --area-mode "$AREA_MODE" \
            --use-rg --rg-mode "$RG_MODE" \
            --use-rdf --rdf-mode "$RDF_MODE" \
            2>&1 | tee "log/extreme_run${RUN_NUM}_len${LENGTH}_A${AREA_MODE}_R${RG_MODE}_D${RDF_MODE}.log"

        # Save the CSV file with a unique name for this run (before it gets overwritten)
        if [ -f "log/flex_len${LENGTH}.csv" ]; then
            cp "log/flex_len${LENGTH}.csv" "log/run${RUN_NUM}_candidates.csv"
        fi

        echo
    done
done

echo "=========================================="
echo "All 8 optimization runs complete!"
echo "Logs saved to log/extreme_run*.log"
echo "=========================================="
echo

# Extract optimized polymers and create extreme test set
echo "Collecting results into extreme_test_set.txt..."
EXTREME_FILE="extreme_test_set.txt"

# Clear/create the file
> "$EXTREME_FILE"

# Extract the polymer from each log file
for i in {1..8}; do
    LOGFILE=$(ls log/extreme_run${i}_*.log 2>/dev/null | head -n 1)
    if [ -f "$LOGFILE" ]; then
        # Extract the line containing "Polymer: " and get just the polymer string
        POLYMER=$(grep "^Polymer:" "$LOGFILE" | sed 's/^Polymer: //')
        if [ -n "$POLYMER" ]; then
            echo "$POLYMER" >> "$EXTREME_FILE"
        fi
    fi
done

echo "=========================================="
echo "Extreme test set saved to: $EXTREME_FILE"
echo "Contains $(wc -l < "$EXTREME_FILE") optimized architectures"
echo "=========================================="
echo

# Extract top 10 candidates from each run
echo "Extracting top 10 candidates from each run..."
TOP_CANDIDATES_FILE="log/top_candidates_per_run.txt"

> "$TOP_CANDIDATES_FILE"

python3 << 'PYTHON_EOF'
import csv
from pathlib import Path

objectives = [
    "max area, max rg, max rdf",
    "max area, max rg, min rdf",
    "max area, min rg, max rdf",
    "max area, min rg, min rdf",
    "min area, max rg, max rdf",
    "min area, max rg, min rdf",
    "min area, min rg, max rdf",
    "min area, min rg, min rdf"
]

output_lines = []

for run_num in range(1, 9):
    csv_file = Path(f"log/run{run_num}_candidates.csv")

    if not csv_file.exists():
        output_lines.append(f"\nRun {run_num}: {objectives[run_num-1]}")
        output_lines.append("  CSV file not found!\n")
        continue

    # Read all candidates
    candidates = []
    seen_polymers = set()

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            polymer = row['polymer']
            try:
                score = float(row['score'])
            except (ValueError, KeyError):
                continue

            # Only keep unique polymers
            if polymer not in seen_polymers:
                seen_polymers.add(polymer)
                candidates.append({
                    'polymer': polymer,
                    'score': score
                })

    # Sort by score (lower is better)
    candidates.sort(key=lambda x: x['score'])

    # Take top 10
    top_10 = candidates[:10]

    output_lines.append(f"\nRun {run_num}: {objectives[run_num-1]}")
    output_lines.append(f"  (Total unique candidates: {len(candidates)})")

    for i, candidate in enumerate(top_10, 1):
        output_lines.append(f"  {i}. {candidate['polymer']}")
        output_lines.append(f"     Score: {candidate['score']:.6f}")

    output_lines.append("")

# Write to file
with open('log/top_candidates_per_run.txt', 'w') as f:
    f.write('\n'.join(output_lines))

print('\n'.join(output_lines))
PYTHON_EOF

echo "=========================================="
echo "Top candidates saved to: $TOP_CANDIDATES_FILE"
echo "=========================================="
