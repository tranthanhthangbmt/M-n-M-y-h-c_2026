import os
import glob

docs_dir = r"d:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Máy học_2026\machineLearningWeb\docs"
for md_file in glob.glob(os.path.join(docs_dir, "chuong_*.md")):
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace the incorrect relative path
    new_content = content.replace("../../Figures/", "../Figures/")
    
    if new_content != content:
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Fixed {os.path.basename(md_file)}")

print("All done!")
