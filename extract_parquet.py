from datasets import load_dataset
import soundfile as sf
import os

print("Loading dataset from Hugging Face...")
dataset = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k")

def save_split(split_name, data):
    base_dir = f"VoiceBank_DEMAND_16k_wav/{split_name}"
    os.makedirs(f"{base_dir}/clean", exist_ok=True)
    os.makedirs(f"{base_dir}/noisy", exist_ok=True)

    for i, s in enumerate(data):
        sf.write(f"{base_dir}/clean/{i:05d}.wav", s["clean"]["array"], 16000)
        sf.write(f"{base_dir}/noisy/{i:05d}.wav", s["noisy"]["array"], 16000)
        if (i+1) % 100 == 0:
            print(f"  → Processed {i+1} samples for {split_name}")

save_split("train", dataset["train"])
save_split("test", dataset["test"])
print("✅ Conversion complete! WAV files saved in VoiceBank_DEMAND_16k_wav/")