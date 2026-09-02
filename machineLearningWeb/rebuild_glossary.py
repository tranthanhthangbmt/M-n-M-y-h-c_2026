import re

# 1. Đọc nội dung gốc
with open('depPlan/glossaryChapter_01.md', 'r', encoding='utf-8') as f:
    text = f.read()

# 1.1 Xử lý hình ảnh (chạy trước để tránh bị convert thành $$)
text = re.sub(r'([ \t]*)\\\\\[\s*\\text\{(Hình (3-\d+)):\s*(.*?)\}\s*\\\\\]', r'\1![Hình \3: \4](../Figures/CH03/Hinh_\3.png)\n\1<span style="display: block; color: #333; font-style: italic; text-align: center; margin-top: 5px; margin-bottom: 15px;"><b>Hình \3: \4</b></span>\n', text)

# 2. Xử lý Toán học (chuyển \\[...\\] thành $$...$$ và \\(...\\) thành $...$)
text = re.sub(r'\\\\\[(.*?)\\\\\]', r'$$\1$$', text, flags=re.DOTALL)
text = re.sub(r'\\\\\((.*?)\\\\\)', r'$\1$', text, flags=re.DOTALL)

# 3. Phân tách thành các phần
parts = re.split(r'#\s+(PHẦN\s+\d+:.*)', text)

accordion_html = ""
for i in range(1, len(parts), 2):
    title = parts[i].strip()
    content = parts[i+1].strip()
    content = re.sub(r'^###\s+(.*)', r'**\1**', content, flags=re.MULTILINE)
    
    # Sửa lỗi thụt lề 8 khoảng trắng cho thẻ <p>
    content = re.sub(r'^ {8}<p ', '    <p ', content, flags=re.MULTILINE)
    
    accordion_html += f"""
<details>
<summary><b style="font-size:1.2em">{title}</b></summary>
<br>

{content}

---

</details>
"""

# 4. Chèn vào chuong_03.md
with open('docs/chuong_03.md', 'r', encoding='utf-8') as f:
    ch3 = f.read()

injection = f"""
#### ** 📚 Thuật ngữ & Khái niệm **

*Dưới đây là tổng hợp toàn bộ các thuật ngữ, khái niệm cốt lõi, công thức và mã nguồn minh họa trong Chương 3 để bạn tra cứu nhanh.*

{accordion_html}
"""

start_marker = "<!-- tabs:start -->"
start_idx = ch3.find(start_marker)

# Tìm và xóa tab cũ nếu có
old_tab_marker = "#### ** 📚 Thuật ngữ & Khái niệm **"
old_tab_idx = ch3.find(old_tab_marker)
if old_tab_idx != -1:
    # Tìm tab tiếp theo để xóa đoạn giữa
    next_tab_marker = "#### ** 📖 Lý thuyết **"
    next_tab_idx = ch3.find(next_tab_marker, old_tab_idx)
    if next_tab_idx != -1:
        ch3 = ch3[:old_tab_idx] + ch3[next_tab_idx:]
    else:
        # Nếu không có tab tiếp theo thì xóa tới hết tabs:end
        end_idx = ch3.find("<!-- tabs:end -->", old_tab_idx)
        if end_idx != -1:
            ch3 = ch3[:old_tab_idx] + ch3[end_idx:]

# Tìm lại start_idx sau khi đã thay đổi chuỗi
start_idx = ch3.find(start_marker)

if start_idx != -1:
    insert_pos = start_idx + len(start_marker)
    new_ch3 = ch3[:insert_pos] + "\n" + injection + "\n" + ch3[insert_pos:]
    with open('docs/chuong_03.md', 'w', encoding='utf-8') as f:
        f.write(new_ch3)
    print("Injected glossary tab successfully.")
else:
    print("Could not find <!-- tabs:start --> in chuong_03.md")
