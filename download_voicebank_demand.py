import os
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

def save_audio_files(dataset_split, split_name, output_dir):
    """
    Hàm duyệt qua từng sample trong dataset và lưu file .wav
    """
    # Các cột chứa dữ liệu audio trong dataset này thường là 'noisy' và 'clean'
    # Nếu dataset có cấu trúc khác, bạn có thể print(dataset_split[0]) để kiểm tra key.
    audio_keys = ['noisy', 'clean']

    print(f"🚀 Đang xử lý tập dữ liệu: {split_name}...")

    for i, item in tqdm(enumerate(dataset_split), total=len(dataset_split), desc=f"Extracting {split_name}"):
        for key in audio_keys:
            if key in item:
                # Tạo đường dẫn thư mục: output/split/type (ví dụ: data/test/clean)
                save_folder = os.path.join(output_dir, split_name, key)
                os.makedirs(save_folder, exist_ok=True)

                # Lấy thông tin audio
                audio_data = item[key]['array']
                sample_rate = item[key]['sampling_rate']

                # Tạo tên file. Vì HF dataset thường không giữ tên file gốc trong object audio,
                # ta dùng index hoặc ID nếu có. Dataset này thường không có cột filename gốc rõ ràng
                # trong object audio, nên ta đặt tên theo format: file_{index}.wav
                # Tuy nhiên, nếu cột 'fileid' hoặc tương tự tồn tại, ta sẽ dùng nó.
                # (Kiểm tra dataset này thường không có sẵn file ID ở lớp ngoài cùng, nên dùng index cho an toàn)
                filename = f"file_{i:05d}.wav"

                # Lưu file
                file_path = os.path.join(save_folder, filename)
                sf.write(file_path, audio_data, sample_rate)

def main():
    # 1. Cấu hình
    DATASET_ID = "JacobLinCool/VoiceBank-DEMAND-16k"
    OUTPUT_DIR = "./voicebank_demand_16k_extracted"

    print(f"📥 Đang tải dataset từ Hugging Face: {DATASET_ID}...")

    # 2. Download và Load dataset (sẽ được cache tự động bởi thư viện datasets)
    try:
        # Load toàn bộ các split (train, test)
        dataset = load_dataset(DATASET_ID)
    except Exception as e:
        print(f"❌ Lỗi khi tải dataset: {e}")
        return

    print("✅ Đã tải dataset thành công!")
    print(f"📂 Dữ liệu sẽ được trích xuất ra thư mục: {os.path.abspath(OUTPUT_DIR)}")
    print("-" * 50)

    # 3. Duyệt qua các split (thường là 'train' và 'test') và lưu file
    for split in dataset.keys():
        save_audio_files(dataset[split], split, OUTPUT_DIR)

    print("-" * 50)
    print("🎉 Hoàn tất! Kiểm tra thư mục output.")

if __name__ == "__main__":
    main()