import os
import shutil
import re

source_dir = r"d:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Máy học_2026\Tài liệu"
target_base_dir = r"d:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Máy học_2026\machineLearningWeb\Figures"

print("Bắt đầu sao chép ảnh...")

copied_chapters = []

for item in os.listdir(source_dir):
    item_path = os.path.join(source_dir, item)
    if os.path.isdir(item_path):
        match = re.search(r"CHƯƠNG\s*(\d+)_files", item, re.IGNORECASE)
        if match:
            chapter_num = int(match.group(1))
            if 9 <= chapter_num <= 19:
                ch_folder_name = f"CH{chapter_num:02d}"
                target_dir = os.path.join(target_base_dir, ch_folder_name)
                
                os.makedirs(target_dir, exist_ok=True)
                
                # copy images
                image_exts = {'.png', '.jpg', '.jpeg', '.gif'}
                copied = 0
                for file in os.listdir(item_path):
                    if os.path.splitext(file)[1].lower() in image_exts:
                        src_file = os.path.join(item_path, file)
                        dst_file = os.path.join(target_dir, file)
                        shutil.copy2(src_file, dst_file)
                        copied += 1
                
                print(f"Đã sao chép {copied} ảnh từ {item} vào thư mục {ch_folder_name}")
                copied_chapters.append(chapter_num)

missing_chapters = set(range(9, 20)) - set(copied_chapters)
if missing_chapters:
    print("\nLưu ý: Không tìm thấy thư mục _files cho các chương:", sorted(list(missing_chapters)))
    print("Có thể các chương này không có ảnh, hoặc bạn đã lưu nhầm định dạng (ví dụ: lưu thành Single File Web Page thay vì Web Page).")
