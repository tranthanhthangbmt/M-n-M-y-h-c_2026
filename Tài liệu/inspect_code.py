from bs4 import BeautifulSoup
import re

html_file = r"d:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Máy học_2026\Tài liệu\scratch\extracted.html"
with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
    soup = BeautifulSoup(f, 'html.parser')

for tag in soup.find_all(True):
    if tag.text and "import matplotlib.pyplot as plt" in tag.text:
        print(f"Tag: {tag.name}")
        print(f"Attrs: {tag.attrs}")
        print(f"Text: {repr(tag.text)}")
        print("-------")
