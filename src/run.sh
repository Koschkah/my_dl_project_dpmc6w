#!/bin/bash
set -e

echo "✅ Docker environment working!"

echo "Running data processing..."
python 01_data_processing.py

echo "Running model training..."
python 02_train.py

echo "Running evaluation..."
python 03_evaluation.py

echo "Pipeline finished successfully."
