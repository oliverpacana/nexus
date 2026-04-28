#!/bin/bash

echo "📊 Counting lines of code..."
echo "---------------------------"

# Find all files, ignore .git and target, count lines, sort by size
find . -type f \
    -not -path "*/\.git/*" \
    -not -path "*/target/*" \
    -exec wc -l {} + | sort -n

echo "---------------------------"
echo "Done!"
