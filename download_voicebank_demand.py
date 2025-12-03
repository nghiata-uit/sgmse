import os
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm
import gc


def save_audio_files_stream(dataset_stream, split_name, output_dir):
    """
    Xử lý dataset theo dạng stream (từng file một) để không tốn RAM.
    """
    audio_keys = ['noisy', 'clean']

    # Tạo thư mục trước
    for key in audio_keys:
        os.makedirs(os.path.join(output_dir, split_name, key), exist_ok=True)

    print(f"🚀 Đang xử lý tập dữ liệu: {split_name} (Streaming Mode)...")

    # Streaming dataset không có hàm len(), nên ta dùng biến đếm thủ công
    count = 0

    # Duyệt qua từng item trong stream
    for item in tqdm(dataset_stream, desc=f"Extracting {split_name}"):
        try:
            for key in audio_keys:
                if key in item:
                    # Lấy thông tin audio
                    audio_data = item[key]['array']
                    sample_rate = item[key]['sampling_rate']

                    # Tạo tên file theo index
                    filename = f"file_{count:05d}.wav"

                    # Lưu file
                    save_path = os.path.join(output_dir, split_name, key, filename)
                    sf.write(save_path, audio_data, sample_rate)

            count += 1

            # Giải phóng bộ nhớ RAM định kỳ mỗi 1000 files (phòng hờ)
            if count % 1000 == 0:
                gc.collect()

        except Exception as e:
            print(f"⚠️ Lỗi ở file thứ {count}: {e}")
            continue

    print(f"✅ Đã trích xuất xong {count} files cho tập {split_name}.")


def main():
    DATASET_ID = "JacobLinCool/VoiceBank-DEMAND-16k"
    OUTPUT_DIR = "./voicebank_demand_16k_extracted"

    print(f"📥 Đang kết nối tới Hugging Face (Streaming Mode): {DATASET_ID}...")

    try:
        # QUAN TRỌNG: streaming=True giúp tải từng phần, KHÔNG tải hết vào RAM
        dataset = load_dataset(DATASET_ID, streaming=True)
    except Exception as e:
        print(f"❌ Lỗi kết nối: {e}")
        return

    print("✅ Kết nối thành công! Bắt đầu trích xuất...")
    print(f"📂 Output dir: {os.path.abspath(OUTPUT_DIR)}")
    print("-" * 50)

    # Duyệt qua các split (train, test) có trong dataset
    for split in dataset.keys():
        save_audio_files_stream(dataset[split], split, OUTPUT_DIR)

    print("-" * 50)
    print("🎉 Hoàn tất! Bạn có thể kiểm tra dung lượng thư mục.")


if __name__ == "__main__":
    main()