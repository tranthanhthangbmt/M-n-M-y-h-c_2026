import os
import re
import urllib.parse
from bs4 import BeautifulSoup
import base64

html_path = r'd:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Máy học_2026\Tài liệu\CHƯƠNG 14.htm'
base_dir = os.path.dirname(html_path)
files_dir = os.path.join(base_dir, 'CHƯƠNG 14_files')
out_dir = r'd:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Máy học_2026\machineLearningWeb\Figures\CH14'

os.makedirs(out_dir, exist_ok=True)

with open(html_path, 'r', encoding='utf-8') as f:
    soup = BeautifulSoup(f, 'html.parser')

images = soup.find_all('img')
extracted_count = 0

for img in images:
    # Find caption
    parent = img.parent
    caption_text = ""
    while parent:
        text = parent.get_text(strip=True)
        if re.search(r'Hình\s*14-\d+', text, re.IGNORECASE):
            caption_text = text
            break
        
        # Check next sibling
        next_sib = parent.find_next_sibling()
        if next_sib:
            sib_text = next_sib.get_text(strip=True)
            if re.search(r'Hình\s*14-\d+', sib_text, re.IGNORECASE):
                caption_text = sib_text
                break
                
        parent = parent.parent

    match = re.search(r'Hình\s*(14-\d+[a-z]?)', caption_text, re.IGNORECASE)
    if match:
        hinh_id = match.group(1).replace(' ', '')
        out_name = f"Hình_{hinh_id}"
        
        src = img.get('src', '')
        if src.startswith('data:image'):
            # Base64
            try:
                header, encoded = src.split(",", 1)
                ext = header.split(';')[0].split('/')[1]
                data = base64.b64decode(encoded)
                out_path = os.path.join(out_dir, f"{out_name}.{ext}")
                with open(out_path, 'wb') as out_f:
                    out_f.write(data)
                extracted_count += 1
                print(f"Extracted {out_name}.{ext} from base64")
            except Exception as e:
                print(f"Error parsing base64 for {out_name}: {e}")
        else:
            # File
            src_unquoted = urllib.parse.unquote(src)
            img_path = os.path.join(base_dir, src_unquoted)
            if os.path.exists(img_path):
                ext = os.path.splitext(img_path)[1]
                if not ext: ext = '.jpg'
                out_path = os.path.join(out_dir, f"{out_name}{ext}")
                import shutil
                shutil.copy2(img_path, out_path)
                extracted_count += 1
                print(f"Copied {out_name}{ext} from {img_path}")
            else:
                print(f"File not found: {img_path}")

print(f"Total extracted: {extracted_count}")
