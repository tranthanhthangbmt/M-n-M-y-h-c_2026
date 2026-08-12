import os
import shutil
import urllib.parse
from bs4 import BeautifulSoup
import re

html_path = r'd:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Máy học_2026\Tài liệu\CHƯƠNG 13.htm'
ch13_files_dir = r'd:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Máy học_2026\Tài liệu\CHƯƠNG 13_files'
ch13_dest_dir = r'd:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Máy học_2026\machineLearningWeb\Figures\CH13'

if not os.path.exists(ch13_dest_dir):
    os.makedirs(ch13_dest_dir)

with open(html_path, 'r', encoding='utf-8') as f:
    html = f.read()

soup = BeautifulSoup(html, 'html.parser')
imgs = soup.find_all('img')

mapping = []
for img in imgs:
    src = img.get('src', '')
    if not src:
        continue
    img_name = src.split('/')[-1]
    
    # Only care about .png or .jpg
    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
        
    p = img.find_parent('p')
    caption_text = ""
    
    # Look at the next few paragraphs
    curr = p
    for _ in range(5):
        if curr:
            curr = curr.find_next_sibling('p')
            if curr:
                text = curr.get_text(separator=' ').strip()
                if re.search(r'Hình\s*13-\d+', text, re.IGNORECASE):
                    caption_text = text
                    break
    
    if caption_text:
        match = re.search(r'(Hình\s*13-\d+)', caption_text, re.IGNORECASE)
        if match:
            hinh_name = match.group(1).replace(' ', '_').replace('\n', '')
            mapping.append((img_name, hinh_name))

print(f"Found {len(mapping)} mappings.")

hinh_to_images = {}
for img_name, hinh_name in mapping:
    if hinh_name not in hinh_to_images:
        hinh_to_images[hinh_name] = []
    hinh_to_images[hinh_name].append(img_name)

final_mapping = {}
for hinh, imgs in hinh_to_images.items():
    # Find the largest image file among imgs
    max_size = 0
    best_img = None
    for img in imgs:
        path = os.path.join(ch13_files_dir, img)
        if not os.path.exists(path):
             path = os.path.join(ch13_files_dir, urllib.parse.unquote(img))
        if os.path.exists(path):
            size = os.path.getsize(path)
            if size > max_size:
                max_size = size
                best_img = urllib.parse.unquote(img)
    if best_img:
        final_mapping[best_img] = hinh

print("Final mapping:")
for img, hinh in final_mapping.items():
    print(f"{img} -> {hinh}")
    
    # Rename the file in CH13
    src_path = os.path.join(ch13_files_dir, img)
    ext = os.path.splitext(img)[1]
    dst_path = os.path.join(ch13_dest_dir, hinh + ext)
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)

print("Done copying renamed images.")
