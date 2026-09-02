import re
with open('docs/chuong_03_temp.md', 'r', encoding='utf-8') as f:
    text = f.read()
print('$$:', len(re.findall(r'\$\$', text)))
