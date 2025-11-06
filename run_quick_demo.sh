#!/bin/bash
set -e
echo ">>> Converting .parquet to .wav..."
pip install -q soundfile pyarrow tqdm
python3 extract_parquet.py
echo "✅ Conversion finished!"
