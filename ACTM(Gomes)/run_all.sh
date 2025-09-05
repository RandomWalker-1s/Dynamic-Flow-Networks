#!/bin/bash
for f in ACTM_example_*.py; do
    echo "Running $f"
    python3 "$f"
done