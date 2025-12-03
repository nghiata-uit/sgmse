"""
Script để download VoiceBank-DEMAND-16k dataset từ HuggingFace
"""

import os
from datasets import load_dataset
import soundfile as sf
from tqdm import tqdm
import numpy as np


    """
    Download VoiceBank-DEMAND-16k dataset và extract thành .wav files

    Args:
        output_dir: Thư mục lưu dataset
    """

    print("=" * 70)
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
            dataset = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k")

        print(f"  Available splits: {list(dataset.keys())}")

        # Process từng split
        for split in dataset.keys():
            print(f"\n{'=' * 70}")
            print(f"PROCESSING {split.upper()} SET")
            print(f"{'=' * 70}")

            clean_dir = os.path.join(output_dir, split, 'clean')

                    # Tạo filename
                    if 'speaker_id' in sample and 'utterance_id' in sample:
                        filename = f"p{sample['speaker_id']}_{sample['utterance_id']:03d}.wav"
                    else:
                        filename = f"sample_{idx:05d}.wav"

                        output_path = os.path.join(clean_dir, filename)
                        sf.write(output_path, clean_audio, clean_sr)


                        noisy_sr = sample['noisy']['sampling_rate']

                        output_path = os.path.join(noisy_dir, filename)
                        sf.write(output_path, noisy_audio, noisy_sr)


        # Print summary


    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Kiểm tra internet connection")
        print("2. Cài đặt required packages:")
        raise


    """In ra thông tin tổng quan về dataset"""

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

        clean_dir = os.path.join(output_dir, split, 'clean')
        noisy_dir = os.path.join(output_dir, split, 'noisy')


        print(f"  Location: {output_dir}/{split}/")

    print("\n" + "=" * 70)
    print("DIRECTORY STRUCTURE:")
    print("=" * 70)
    print(f"{output_dir}/")
            print(f"├── {split}/")

    print("\n✅ DOWNLOAD AND EXTRACTION COMPLETED!")
    print(f"📁 All files saved to: {os.path.abspath(output_dir)}")


def verify_audio_files(output_dir, num_samples=3):
    """
    Verify một số audio files để đảm bảo đã extract đúng
    """
    print("\n" + "=" * 70)
    print("VERIFYING AUDIO FILES...")
    print("=" * 70)

    for split in ['train', 'test']:
        clean_dir = os.path.join(output_dir, split, 'clean')
        noisy_dir = os.path.join(output_dir, split, 'noisy')

        clean_files = sorted([f for f in os.listdir(clean_dir) if f.endswith('.wav')])[:num_samples]

        for filename in clean_files:
            clean_path = os.path.join(clean_dir, filename)
            noisy_path = os.path.join(noisy_dir, filename)


                print(f"\n  File: {filename}")
                print(f"    ✓ Files verified")


if __name__ == "__main__":
    # Download và extract dataset
    output_directory = "./voicebank_demand_16k"


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