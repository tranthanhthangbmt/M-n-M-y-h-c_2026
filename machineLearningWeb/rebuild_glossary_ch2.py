import re

# 1. Đọc nội dung gốc
with open('depPlan/glossaryChapter_02.md', 'r', encoding='utf-8') as f:
    text = f.read()

# 1.0 Xóa các trích dẫn [cite: ...]
text = re.sub(r'(?:\\\\)*[ \t]*\[cite: \d+\]', '', text)

# 1.1 Xử lý hình ảnh (chạy trước để tránh bị convert thành $$)
# Dành riêng cho Chương 2 (Hình 2-X)
text = re.sub(r'([ \t]*)\\\\\[\s*\\text\{(Hình (2-\d+)):\s*(.*?)\}\s*\\\\\]', r'\1![Hình \3: \4](../Figures/CH02/Hinh_\3.png)\n\1<span style="display: block; color: #333; font-style: italic; text-align: center; margin-top: 5px; margin-bottom: 15px;"><b>Hình \3: \4</b></span>\n', text)

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

# 4. Chèn vào chuong_02.md
with open('docs/chuong_02.md', 'r', encoding='utf-8') as f:
    ch2 = f.read()

injection = f"""
#### ** 📚 Thuật ngữ & Khái niệm **

*Dưới đây là tổng hợp toàn bộ các thuật ngữ, khái niệm cốt lõi, công thức và mã nguồn minh họa trong Chương 2 để bạn tra cứu nhanh.*

{accordion_html}
"""

start_marker = "<!-- tabs:start -->"
start_idx = ch2.find(start_marker)

# Tìm và xóa tab cũ nếu có
old_tab_marker = "#### ** 📚 Thuật ngữ & Khái niệm **"
old_tab_idx = ch2.find(old_tab_marker)
if old_tab_idx != -1:
    # Tìm tab tiếp theo để xóa đoạn giữa
    next_tab_marker = "#### ** 📖 Lý thuyết **"
    next_tab_idx = ch2.find(next_tab_marker, old_tab_idx)
    if next_tab_idx != -1:
        ch2 = ch2[:old_tab_idx] + ch2[next_tab_idx:]
    else:
        # Nếu không có tab tiếp theo thì xóa tới hết tabs:end
        end_idx = ch2.find("<!-- tabs:end -->", old_tab_idx)
        if end_idx != -1:
            ch2 = ch2[:old_tab_idx] + ch2[end_idx:]

# Tìm lại start_idx sau khi đã thay đổi chuỗi
start_idx = ch2.find(start_marker)

if start_idx != -1:
    insert_pos = start_idx + len(start_marker)
    new_ch2 = ch2[:insert_pos] + "\n" + injection + "\n" + ch2[insert_pos:]
    with open('docs/chuong_02.md', 'w', encoding='utf-8') as f:
        f.write(new_ch2)
    print("Injected glossary tab successfully into chuong_02.md")
else:
    print("Could not find <!-- tabs:start --> in chuong_02.md")
