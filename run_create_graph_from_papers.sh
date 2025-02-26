#!/usr/bin/env bash

# Generate a timestamp
timestamp=$(date +"%Y%m%d_%H%M%S")

# Run the script and save the output to a file with the timestamp
python create_graph_from_papers.py > "graph_output_${timestamp}.txt" 2>&1
echo "Output saved to graph_output_${timestamp}.txt"