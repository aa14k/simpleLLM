#!/bin/bash

# Check if at least one file is provided
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 moeGPT/train.py moeGPT/model.py moeGPT/utils.py"
    exit 1
fi

# Loop through all arguments
for file in "$@"; do
    if [ -f "$file" ]; then
        echo "----$file----"
        echo ""
        cat "$file"
        echo ""
        echo "" # Add extra spacing between files
    else
        echo "Warning: File '$file' not found." >&2
    fi
done