import os
import soundfile as sf
from datasets import load_dataset

def save_audio(split_name, dataset_split):
    os.makedirs(f"data/demo/{split_name}/noisy", exist_ok=True)
    os.makedirs(f"data/demo/{split_name}/clean", exist_ok=True)

    for i, sample in enumerate(dataset_split):
        noisy_audio = sample["noisy"]["array"]
        clean_audio = sample["clean"]["array"]

        # Save both audio files (16 kHz)
        sf.write(f"data/demo/{split_name}/noisy/{i:05d}.wav", noisy_audio, 16000)
        sf.write(f"data/demo/{split_name}/clean/{i:05d}.wav", clean_audio, 16000)

        if i % 1000 == 0:
            print(f"Saved {i} samples from {split_name}")

dataset = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k")
# Save both splits
save_audio("train", dataset["train"])
save_audio("valid", dataset["test"])
