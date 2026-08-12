import os
from bs4 import BeautifulSoup

html_file = r"d:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Máy học_2026\Tài liệu\scratch\extracted.html"

with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
    soup = BeautifulSoup(f, 'html.parser')

print("H1 tags:")
for i, h1 in enumerate(soup.find_all('h1')):
    print(f"H1 {i}: {h1.text.strip()}")
    if i > 20:
        break

print("\nH2 tags:")
for i, h2 in enumerate(soup.find_all('h2')):
    print(f"H2 {i}: {h2.text.strip()}")
    if i > 20:
        break

print("\nImage sources:")
for i, img in enumerate(soup.find_all('img')):
    print(f"Img {i}: {img.get('src')}")
    if i > 20:
        break
