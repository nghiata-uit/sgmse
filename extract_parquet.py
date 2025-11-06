#!/usr/bin/env python3
"""
Convert .parquet audio files (VoiceBank-DEMAND-16k format) to .wav
Safely processes large files without running out of memory.
"""

import os
import soundfile as sf
import pyarrow.parquet as pq
from tqdm import tqdm

# === Configuration ===
PARQUET_DIR = "./VoiceBank_DEMAND_16k/data"   # Folder containing parquet files
OUTPUT_DIR = "./VoiceBank_DEMAND_16k_wav"     # Where to save .wav files
SAMPLE_RATE = 16000                           # Hz

# === Function to convert a single parquet file ===
def convert_parquet_to_wav(parquet_path, split_name="train"):
    reader = pq.ParquetFile(parquet_path)
    num_groups = reader.metadata.num_row_groups

    out_clean = os.path.join(OUTPUT_DIR, split_name, "clean")
    out_noisy = os.path.join(OUTPUT_DIR, split_name, "noisy")
    os.makedirs(out_clean, exist_ok=True)
    os.makedirs(out_noisy, exist_ok=True)

    print(f"Processing {parquet_path} ({num_groups} row groups)...")

    counter = 0
    # Process in chunks (row groups) to avoid memory overflow
    for rg in range(num_groups):
        table = reader.read_row_group(rg)
        arrays = table.to_pydict()

        # Ensure both 'clean' and 'noisy' exist
        clean_audio = arrays.get("clean", [])
        noisy_audio = arrays.get("noisy", [])

        for i in tqdm(range(len(clean_audio)), desc=f"RowGroup {rg+1}/{num_groups}", leave=False):
            clean = clean_audio[i]["array"]
            noisy = noisy_audio[i]["array"]

            sf.write(os.path.join(out_clean, f"{counter:07d}.wav"), clean, SAMPLE_RATE)
            sf.write(os.path.join(out_noisy, f"{counter:07d}.wav"), noisy, SAMPLE_RATE)
            counter += 1

    print(f"✅ Done: {parquet_path} → {counter} files saved")

# === Main script ===
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for file in os.listdir(PARQUET_DIR):
        if file.endswith(".parquet"):
            split = "train" if "train" in file.lower() else "test"
            convert_parquet_to_wav(os.path.join(PARQUET_DIR, file), split)
