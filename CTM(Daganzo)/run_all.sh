#!/bin/bash
for f in CTM_example_*.py; do
    echo "Running $f"
    python3 "$f"
done