"""
Script để download VoiceBank-DEMAND-16k dataset từ HuggingFace
và extract các file audio thành .wav
"""

import os
from datasets import load_dataset
import soundfile as sf
from tqdm import tqdm
import numpy as np


def download_and_extract_voicebank_demand(output_dir="./voicebank_demand_16k"):
    """
    Download VoiceBank-DEMAND-16k dataset và extract thành .wav files

    Args:
        output_dir: Thư mục lưu dataset
    """

    print("=" * 70)
    print("DOWNLOADING VOICEBANK-DEMAND-16k DATASET")
    print("=" * 70)

    # Tạo thư mục output
    os.makedirs(output_dir, exist_ok=True)

    # Tạo các thư mục con
    splits = ['train', 'test']
    audio_types = ['clean', 'noisy']

    for split in splits:
        for audio_type in audio_types:
            path = os.path.join(output_dir, split, audio_type)
            os.makedirs(path, exist_ok=True)
            print(f"✓ Created directory: {path}")

    print("\n" + "=" * 70)
    print("LOADING DATASET FROM HUGGINGFACE...")
    print("=" * 70)

    try:
        # Load dataset từ HuggingFace
        dataset = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k")

        print(f"\n✓ Dataset loaded successfully!")
        print(f"  Available splits: {list(dataset.keys())}")

        # Process từng split
        for split in dataset.keys():
            print(f"\n{'=' * 70}")
            print(f"PROCESSING {split.upper()} SET")
            print(f"{'=' * 70}")
            print(f"Total samples: {len(dataset[split])}")

            # Extract clean audio
            print(f"\n[1/2] Extracting CLEAN audio...")
            clean_dir = os.path.join(output_dir, split, 'clean')
            for idx, sample in enumerate(tqdm(dataset[split], desc=f"Clean {split}")):
                # Get clean audio
                clean_audio = sample['clean']['array']
                clean_sr = sample['clean']['sampling_rate']

                # Tạo filename
                if 'speaker_id' in sample and 'utterance_id' in sample:
                    filename = f"p{sample['speaker_id']}_{sample['utterance_id']:03d}.wav"
                else:
                    filename = f"sample_{idx:05d}.wav"

                # Save as .wav
                output_path = os.path.join(clean_dir, filename)
                sf.write(output_path, clean_audio, clean_sr)

            print(f"  ✓ Saved {len(dataset[split])} clean files to: {clean_dir}")

            # Extract noisy audio
            print(f"\n[2/2] Extracting NOISY audio...")
            noisy_dir = os.path.join(output_dir, split, 'noisy')
            for idx, sample in enumerate(tqdm(dataset[split], desc=f"Noisy {split}")):
                # Get noisy audio
                noisy_audio = sample['noisy']['array']
                noisy_sr = sample['noisy']['sampling_rate']

                # Tạo filename (giống với clean)
                if 'speaker_id' in sample and 'utterance_id' in sample:
                    filename = f"p{sample['speaker_id']}_{sample['utterance_id']:03d}.wav"
                else:
                    filename = f"sample_{idx:05d}.wav"

                # Save as .wav
                output_path = os.path.join(noisy_dir, filename)
                sf.write(output_path, noisy_audio, noisy_sr)

            print(f"  ✓ Saved {len(dataset[split])} noisy files to: {noisy_dir}")

        # Print summary
        print_dataset_summary(output_dir, dataset)

        return dataset

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Kiểm tra internet connection")
        print("2. Cài đặt required packages:")
        print("   pip install datasets soundfile tqdm librosa")
        print("3. Kiểm tra HuggingFace dataset availability")
        raise


def print_dataset_summary(output_dir, dataset):
    """In ra thông tin tổng quan về dataset"""

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    for split in dataset.keys():
        print(f"\n{split.upper()} SET:")
        print(f"  Total samples: {len(dataset[split])}")

        # Get first sample để check info
        sample = dataset[split][0]
        clean_sr = sample['clean']['sampling_rate']
        clean_duration = len(sample['clean']['array']) / clean_sr

        print(f"  Sampling rate: {clean_sr} Hz")
        print(f"  Example duration: {clean_duration:.2f} seconds")

        # Check file structure
        clean_dir = os.path.join(output_dir, split, 'clean')
        noisy_dir = os.path.join(output_dir, split, 'noisy')

        clean_files = len([f for f in os.listdir(clean_dir) if f.endswith('.wav')])
        noisy_files = len([f for f in os.listdir(noisy_dir) if f.endswith('.wav')])

        print(f"  Clean files saved: {clean_files}")
        print(f"  Noisy files saved: {noisy_files}")
        print(f"  Location: {output_dir}/{split}/")

    print("\n" + "=" * 70)
    print("DIRECTORY STRUCTURE:")
    print("=" * 70)
    print(f"{output_dir}/")
    for split in dataset.keys():
        print(f"├── {split}/")
        print(f"│   ├── clean/  ({len(dataset[split])} .wav files)")
        print(f"│   └── noisy/  ({len(dataset[split])} .wav files)")

    print("\n✅ DOWNLOAD AND EXTRACTION COMPLETED!")
    print(f"📁 All files saved to: {os.path.abspath(output_dir)}")


def verify_audio_files(output_dir, num_samples=3):
    """
    Verify một số audio files để đảm bảo đã extract đúng
    """
    print("\n" + "=" * 70)
    print("VERIFYING AUDIO FILES...")
    print("=" * 70)

    import librosa

    for split in ['train', 'test']:
        print(f"\n{split.upper()} SET:")

        clean_dir = os.path.join(output_dir, split, 'clean')
        noisy_dir = os.path.join(output_dir, split, 'noisy')

        clean_files = sorted([f for f in os.listdir(clean_dir) if f.endswith('.wav')])[:num_samples]

        for filename in clean_files:
            clean_path = os.path.join(clean_dir, filename)
            noisy_path = os.path.join(noisy_dir, filename)

            # Load audio
            clean_audio, clean_sr = librosa.load(clean_path, sr=None)
            noisy_audio, noisy_sr = librosa.load(noisy_path, sr=None)

            print(f"\n  File: {filename}")
            print(f"    Clean: {len(clean_audio)} samples, {clean_sr} Hz, {len(clean_audio) / clean_sr:.2f}s")
            print(f"    Noisy: {len(noisy_audio)} samples, {noisy_sr} Hz, {len(noisy_audio) / noisy_sr:.2f}s")
            print(f"    ✓ Files verified")


if __name__ == "__main__":
    # Download và extract dataset
    output_directory = "./voicebank_demand_16k"

    dataset = download_and_extract_voicebank_demand(output_directory)

    # Verify audio files
    verify_audio_files(output_directory, num_samples=3)

    print("\n" + "=" * 70)
    print("READY TO USE!")
    print("=" * 70)
    print("\nBạn có thể sử dụng dataset với:")
    print(f"  Clean train audio: {output_directory}/train/clean/")
    print(f"  Noisy train audio: {output_directory}/train/noisy/")
    print(f"  Clean test audio:  {output_directory}/test/clean/")
    print(f"  Noisy test audio:  {output_directory}/test/noisy/")