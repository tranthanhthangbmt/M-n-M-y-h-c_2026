import re

# 1. Đọc nội dung gốc
with open('depPlan/glossaryChapter_01.md', 'r', encoding='utf-8') as f:
    text = f.read()

# 1.0 Xóa các trích dẫn [cite: ...]
text = re.sub(r'(?:\\\\)*[ \t]*\[cite: \d+\]', '', text)

import os

def replace_image(match):
    indent = match.group(1)
    full_hinh_text = match.group(2) # e.g. Hình 1-1
    hinh_id = match.group(3) # e.g. 1-1
    caption = match.group(4)
    
    # Check extension
    ext = 'png'
    if os.path.exists(f'Figures/CH01/Hinh_{hinh_id}.jpg'):
        ext = 'jpg'
    elif os.path.exists(f'Figures/CH01/Hinh_{hinh_id}.jpeg'):
        ext = 'jpeg'
        
    img_markdown = f'{indent}![Hình {hinh_id}: {caption}](../Figures/CH01/Hinh_{hinh_id}.{ext})'
    span_html = f'{indent}<span style="display: block; color: #333; font-style: italic; text-align: center; margin-top: 5px; margin-bottom: 15px;"><b>Hình {hinh_id}: {caption}</b></span>'
    
    return f'{img_markdown}\n{span_html}\n'

# 1.1 Xử lý hình ảnh (chạy trước để tránh bị convert thành $$)
# Dành riêng cho Chương 1 (Hình 1-X)
text = re.sub(r'([ \t]*)\\\\\[\s*\\text\{(Hình (1-\d+)):\s*(.*?)\}\s*\\\\\]', replace_image, text)

# 2. Xử lý Toán học (chuyển \\[...\\] thành $$...$$ và \\(...\\) thành $...$)
text = re.sub(r'\\\\\[(.*?)\\\\\]', r'$$\1$$', text, flags=re.DOTALL)
text = re.sub(r'\\\\\((.*?)\\\\\)', r'$\1$', text, flags=re.DOTALL)

# 3. Phân tách thành các phần
parts = re.split(r'#\s+(PHẦN\s+\d+:.*)', text)

accordion_html = ""
for i in range(1, len(parts), 2):
    title = parts[i].strip()
    content = parts[i+1].strip()
    
    # Biến các heading H3 thành chữ đậm
    content = re.sub(r'^###\s+(.*)', r'**\1**', content, flags=re.MULTILINE)
    
    accordion_html += f"""
<details>
<summary><b style="font-size:1.2em">{title}</b></summary>
<br>

{content}

---

</details>
"""

# 4. Chèn vào chuong_01.md
with open('docs/chuong_01.md', 'r', encoding='utf-8') as f:
    ch_text = f.read()

injection = f"""
#### ** 📚 Thuật ngữ & Khái niệm **

*Dưới đây là tổng hợp toàn bộ các thuật ngữ, khái niệm cốt lõi, công thức và mã nguồn minh họa trong Chương 1 để bạn tra cứu nhanh.*

{accordion_html}
"""

start_marker = "<!-- tabs:start -->"
start_idx = ch_text.find(start_marker)

# Tìm và xóa tab cũ nếu có
old_tab_marker = "#### ** 📚 Thuật ngữ & Khái niệm **"
old_tab_idx = ch_text.find(old_tab_marker)
if old_tab_idx != -1:
    # Tìm tab tiếp theo để xóa đoạn giữa
    next_tab_marker = "#### ** 📖 Lý thuyết **"
    next_tab_idx = ch_text.find(next_tab_marker, old_tab_idx)
    if next_tab_idx != -1:
        ch_text = ch_text[:old_tab_idx] + ch_text[next_tab_idx:]
    else:
        # Nếu không có tab tiếp theo thì xóa tới hết tabs:end
        end_idx = ch_text.find("<!-- tabs:end -->", old_tab_idx)
        if end_idx != -1:
            ch_text = ch_text[:old_tab_idx] + ch_text[end_idx:]

# Tìm lại start_idx sau khi đã thay đổi chuỗi
start_idx = ch_text.find(start_marker)

if start_idx != -1:
    insert_pos = start_idx + len(start_marker)
    new_ch_text = ch_text[:insert_pos] + "\n" + injection + "\n" + ch_text[insert_pos:]
    with open('docs/chuong_01.md', 'w', encoding='utf-8') as f:
        f.write(new_ch_text)
    print("Injected glossary tab successfully into chuong_01.md")
else:
    print("Could not find <!-- tabs:start --> in chuong_01.md")
