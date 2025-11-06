#!/bin/bash
set -e

# === Step 1: Install dependencies ===
echo ">>> Installing dependencies..."
pip install -U "huggingface_hub[cli]" datasets soundfile pyarrow --quiet

# === Step 2: Download dataset using hf CLI ===
echo ">>> Downloading VoiceBank-DEMAND-16k dataset..."
hf download JacobLinCool/VoiceBank-DEMAND-16k --repo-type dataset --local-dir ./VoiceBank_DEMAND_16k

# === Step 3: Convert .parquet -> .wav ===
echo ">>> Converting parquet files to WAV..."
python extract_parquet.py

echo "✅ Conversion completed successfully!"
