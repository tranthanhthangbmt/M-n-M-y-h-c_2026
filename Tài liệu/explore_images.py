import os
import re
from bs4 import BeautifulSoup

html_file = r"d:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Máy học_2026\Tài liệu\scratch\extracted.html"

with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
    soup = BeautifulSoup(f, 'html.parser')

print("Looking for image tags:")
for i, tag in enumerate(soup.find_all(lambda t: t.name in ['img', 'v:imagedata', 'image'])):
    print(f"Tag: {tag.name}, attrs: {tag.attrs}")
    if i > 20:
        break

print("\nLooking for image captions:")
# Search for paragraphs containing "Hình " or "Figure "
count = 0
for p in soup.find_all(['p', 'div', 'span']):
    text = p.text.strip()
    if re.match(r'^(Hình|Figure)\s+\d+-\d+', text, re.IGNORECASE):
        print(f"Caption: {text[:100]}")
        count += 1
        if count > 20:
            break
