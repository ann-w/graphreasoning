#!/usr/bin/env bash

python create_graph_from_papers.py > graph_output.txt 2>&1
echo "Output saved to graph_output.txt"