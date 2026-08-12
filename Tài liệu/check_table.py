from bs4 import BeautifulSoup

html_file = r"d:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Máy học_2026\Tài liệu\scratch\extracted.html"
with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
    soup = BeautifulSoup(f, 'html.parser')

tables = soup.find_all('table')
print(f"Found {len(tables)} tables.")
if tables:
    t = tables[0]
    for tr in t.find_all('tr'):
        cols = tr.find_all(['th', 'td'])
        print("ROW:", [c.text.strip().replace('\n', ' ') for c in cols])
