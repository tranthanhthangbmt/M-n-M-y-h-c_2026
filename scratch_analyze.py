import json
import sys

with open(r'D:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Máy học_2026\machineLearningWeb\TaiLieu\NotebookJupyter\01_the_machine_learning_landscape.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for i, cell in enumerate(nb['cells']):
    ctype = cell['cell_type']
    if ctype in ('markdown', 'code'):
        source = ''.join(cell.get('source', []))
        short_source = source[:100].replace('\n', ' ')
        print(f"Cell {i:03d} ({ctype:8s}): {short_source}...")
