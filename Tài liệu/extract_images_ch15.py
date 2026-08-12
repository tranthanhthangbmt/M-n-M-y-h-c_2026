import os
import re
import shutil
from bs4 import BeautifulSoup
import urllib.parse

html_file = r'd:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Máy học_2026\Tài liệu\CHƯƠNG 15.htm'
html_dir = os.path.dirname(html_file)
output_dir = r'd:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Máy học_2026\machineLearningWeb\Figures\CH15'
os.makedirs(output_dir, exist_ok=True)

with open(html_file, 'r', encoding='utf-8') as f:
    soup = BeautifulSoup(f.read(), 'html.parser')

images_copied = 0
for img in soup.find_all('img'):
    src = img.get('src')
    if not src:
        continue
    
    # Find the nearest caption
    parent = img.find_parent('p')
    caption = ""
    if parent:
        # Check if this paragraph or the next has the caption
        text = parent.text.strip()
        if re.search(r'Hình 15-\d+', text):
            caption = text
        else:
            next_sib = parent.find_next_sibling('p')
            if next_sib and re.search(r'Hình 15-\d+', next_sib.text.strip()):
                caption = next_sib.text.strip()
    
    if caption:
        match = re.search(r'(Hình 15-\d+)', caption)
        if match:
            fig_name = match.group(1).replace(' ', '_')
            
            src_unquoted = urllib.parse.unquote(src)
            src_path = os.path.join(html_dir, src_unquoted)
            
            if os.path.exists(src_path):
                ext = os.path.splitext(src_path)[1]
                dest_path = os.path.join(output_dir, fig_name + ext)
                shutil.copy2(src_path, dest_path)
                print(f"Copied {fig_name}{ext} from {os.path.basename(src_path)}")
                images_copied += 1
            else:
                print(f"File not found: {src_path}")

print(f"Total images copied: {images_copied}")
