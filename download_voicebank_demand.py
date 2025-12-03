"""
Script để download VoiceBank-DEMAND-16k dataset từ HuggingFace
và extract các file audio thành .wav - OPTIMIZED VERSION
"""

import os
import gc
from datasets import load_dataset
import soundfile as sf
from tqdm import tqdm
import numpy as np


def download_and_extract_voicebank_demand(output_dir="./voicebank_demand_16k", streaming=True):
    """
    Download VoiceBank-DEMAND-16k dataset và extract thành .wav files
    
    Args:
        output_dir: Thư mục lưu dataset
        streaming: Sử dụng streaming mode để tiết kiệm RAM (default: True)
    """
    
    print("=" * 70)
    print("DOWNLOADING VOICEBANK-DEMAND-16k DATASET (OPTIMIZED)")
    print("=" * 70)
    print(f"Streaming mode: {streaming}")
    
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
        # Load dataset với streaming mode để tiết kiệm RAM
        if streaming:
            dataset = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k", streaming=True)
            print(f"\n✓ Dataset loaded in STREAMING mode (memory efficient)")
        else:
            dataset = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k")
            print(f"\n✓ Dataset loaded in NORMAL mode")
        
        print(f"  Available splits: {list(dataset.keys())}")
        
        # Process từng split
        for split in dataset.keys():
            print(f"\n{'=' * 70}")
            print(f"PROCESSING {split.upper()} SET")
            print(f"{'=' * 70}")
            
            clean_dir = os.path.join(output_dir, split, 'clean')
            noisy_dir = os.path.join(output_dir, split, 'noisy')
            
            # Counters
            clean_count = 0
            noisy_count = 0
            error_count = 0
            
            # Process samples one by one (streaming compatible)
            print(f"\nProcessing samples...")
            
            # Tạo progress bar
            pbar = tqdm(desc=f"{split}", unit="samples")
            
            for idx, sample in enumerate(dataset[split]):
                try:
                    # Tạo filename
                    if 'speaker_id' in sample and 'utterance_id' in sample:
                        filename = f"p{sample['speaker_id']}_{sample['utterance_id']:03d}.wav"
                    else:
                        filename = f"sample_{idx:05d}.wav"
                    
                    # Extract và save CLEAN audio
                    try:
                        clean_audio = np.array(sample['clean']['array'], dtype=np.float32)
                        clean_sr = sample['clean']['sampling_rate']
                        
                        output_path = os.path.join(clean_dir, filename)
                        sf.write(output_path, clean_audio, clean_sr)
                        clean_count += 1
                        
                        # Free memory
                        del clean_audio
                        
                    except Exception as e:
                        print(f"\n  ⚠️  Error saving clean file {filename}: {e}")
                        error_count += 1
                    
                    # Extract và save NOISY audio
                    try:
                        noisy_audio = np.array(sample['noisy']['array'], dtype=np.float32)
                        noisy_sr = sample['noisy']['sampling_rate']
                        
                        output_path = os.path.join(noisy_dir, filename)
                        sf.write(output_path, noisy_audio, noisy_sr)
                        noisy_count += 1
                        
                        # Free memory
                        del noisy_audio
                        
                    except Exception as e:
                        print(f"\n  ⚠️  Error saving noisy file {filename}: {e}")
                        error_count += 1
                    
                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({
                        'clean': clean_count,
                        'noisy': noisy_count,
                        'errors': error_count
                    })
                    
                    # Garbage collection mỗi 100 samples
                    if (idx + 1) % 100 == 0:
                        gc.collect()
                    
                except KeyboardInterrupt:
                    print("\n\n⚠️  Download interrupted by user")
                    pbar.close()
                    return None
                    
                except Exception as e:
                    print(f"\n  ⚠️  Error processing sample {idx}: {e}")
                    error_count += 1
                    continue
            
            pbar.close()
            
            print(f"\n  ✓ Saved {clean_count} clean files to: {clean_dir}")
            print(f"  ✓ Saved {noisy_count} noisy files to: {noisy_dir}")
            if error_count > 0:
                print(f"  ⚠️  {error_count} errors encountered")
            
            # Final garbage collection
            gc.collect()
        
        # Print summary
        print_dataset_summary(output_dir)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Kiểm tra internet connection")
        print("2. Cài đặt required packages:")
        print("   pip install datasets soundfile tqdm numpy")
        print("3. Giảm RAM usage bằng cách:")
        print("   - Đóng các ứng dụng khác")
        print("   - Sử dụng streaming=True (đã bật mặc định)")
        print("4. Kiểm tra disk space (cần ~2-3 GB)")
        raise


def print_dataset_summary(output_dir):
    """In ra thông tin tổng quan về dataset"""
    
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    
    for split in ['train', 'test']:
        clean_dir = os.path.join(output_dir, split, 'clean')
        noisy_dir = os.path.join(output_dir, split, 'noisy')
        
        if not os.path.exists(clean_dir):
            continue
            
        print(f"\n{split.upper()} SET:")
        
        clean_files = [f for f in os.listdir(clean_dir) if f.endswith('.wav')]
        noisy_files = [f for f in os.listdir(noisy_dir) if f.endswith('.wav')]
        
        print(f"  Clean files saved: {len(clean_files)}")
        print(f"  Noisy files saved: {len(noisy_files)}")
        print(f"  Location: {output_dir}/{split}/")
        
        # Check first file for details
        if clean_files:
            first_file = os.path.join(clean_dir, clean_files[0])
            try:
                info = sf.info(first_file)
                print(f"  Sampling rate: {info.samplerate} Hz")
                print(f"  Example duration: {info.duration:.2f} seconds")
            except:
                pass
    
    print("\n" + "=" * 70)
    print("DIRECTORY STRUCTURE:")
    print("=" * 70)
    print(f"{output_dir}/")
    for split in ['train', 'test']:
        clean_dir = os.path.join(output_dir, split, 'clean')
        if os.path.exists(clean_dir):
            clean_count = len([f for f in os.listdir(clean_dir) if f.endswith('.wav')])
            noisy_count = len([f for f in os.listdir(os.path.join(output_dir, split, 'noisy')) if f.endswith('.wav')])
            print(f"├── {split}/")
            print(f"│   ├── clean/  ({clean_count} .wav files)")
            print(f"│   └── noisy/  ({noisy_count} .wav files)")
    
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
        
        if not os.path.exists(clean_dir):
            continue
        
        print(f"\n{split.upper()} SET:")
        
        clean_files = sorted([f for f in os.listdir(clean_dir) if f.endswith('.wav')])[:num_samples]
        
        for filename in clean_files:
            clean_path = os.path.join(clean_dir, filename)
            noisy_path = os.path.join(noisy_dir, filename)
            
            try:
                # Load audio info without loading entire file
                clean_info = sf.info(clean_path)
                noisy_info = sf.info(noisy_path)
                
                print(f"\n  File: {filename}")
                print(f"    Clean: {clean_info.frames} samples, {clean_info.samplerate} Hz, {clean_info.duration:.2f}s")
                print(f"    Noisy: {noisy_info.frames} samples, {noisy_info.samplerate} Hz, {noisy_info.duration:.2f}s")
                print(f"    ✓ Files verified")
                
            except Exception as e:
                print(f"    ✗ Error: {e}")


if __name__ == "__main__":
    # Download và extract dataset
    output_directory = "./voicebank_demand_16k"
    
    print("\n💡 TIP: Script này sử dụng streaming mode để tiết kiệm RAM")
    print("         Quá trình có thể mất 10-30 phút tùy tốc độ internet\n")
    
    success = download_and_extract_voicebank_demand(output_directory, streaming=True)
    
    if success:
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
    else:
        print("\n⚠️  Download incomplete. Please try again.")
