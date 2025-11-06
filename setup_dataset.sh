#!/bin/bash

echo "==================================="
echo "VoiceBank-DEMAND Dataset Setup"
echo "==================================="

# Install dependencies
echo "Installing dependencies..."
pip install -q datasets soundfile librosa tqdm numpy

# Run download script
echo "Downloading and extracting dataset..."
python download_voicebank_demand.py

echo ""
echo "==================================="
echo "Setup completed!"
echo "==================================="