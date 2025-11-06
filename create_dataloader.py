"""
DataLoader cho VoiceBank-DEMAND dataset
"""

import torch
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import os
from pathlib import Path
import numpy as np


class VoiceBankDemandDataset(Dataset):
    """
    Dataset class cho VoiceBank-DEMAND
    """

    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir: Thư mục root của dataset
            split: 'train' hoặc 'test'
            transform: Optional transform để apply lên audio
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        self.clean_dir = self.root_dir / split / 'clean'
        self.noisy_dir = self.root_dir / split / 'noisy'

        # Get list of files
        self.clean_files = sorted([f for f in os.listdir(self.clean_dir) if f.endswith('.wav')])
        self.noisy_files = sorted([f for f in os.listdir(self.noisy_dir) if f.endswith('.wav')])

        assert len(self.clean_files) == len(self.noisy_files), \
            "Number of clean and noisy files must match!"

        print(f"Loaded {len(self.clean_files)} samples from {split} set")

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        # Load clean audio
        clean_path = self.clean_dir / self.clean_files[idx]
        clean_audio, sr = sf.read(clean_path)

        # Load noisy audio
        noisy_path = self.noisy_dir / self.noisy_files[idx]
        noisy_audio, _ = sf.read(noisy_path)

        # Convert to tensor
        clean_audio = torch.FloatTensor(clean_audio)
        noisy_audio = torch.FloatTensor(noisy_audio)

        # Apply transform if specified
        if self.transform:
            clean_audio = self.transform(clean_audio)
            noisy_audio = self.transform(noisy_audio)

        return {
            'clean': clean_audio,
            'noisy': noisy_audio,
            'filename': self.clean_files[idx],
            'sr': sr
        }


def create_dataloaders(root_dir, batch_size=16, num_workers=4):
    """
    Tạo train và test dataloaders
    """
    train_dataset = VoiceBankDemandDataset(root_dir, split='train')
    test_dataset = VoiceBankDemandDataset(root_dir, split='test')

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


# Example usage
if __name__ == "__main__":
    root_dir = "./voicebank_demand_16k"

    # Create dataloaders
    train_loader, test_loader = create_dataloaders(root_dir, batch_size=8)

    print(f"\nTrain loader: {len(train_loader)} batches")
    print(f"Test loader: {len(test_loader)} batches")

    # Test loading a batch
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Clean: {batch['clean'].shape}")
    print(f"  Noisy: {batch['noisy'].shape}")
    print(f"  Filenames: {batch['filename']}")
    print(f"  Sample rate: {batch['sr'][0]} Hz")