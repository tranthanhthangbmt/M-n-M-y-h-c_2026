import re

def process_glossary():
    # 1. Read glossary file
    with open('depPlan/glossaryChapter_01.md', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 2. Extract content from PHẦN 1 onwards
    start_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('# PHẦN 1:'):
            start_idx = i
            break
    
    content_lines = lines[start_idx:]
    
    # 3. Process lines
    processed = []
    in_details = False
    
    for line in content_lines:
        if line.startswith('# PHẦN'):
            if in_details:
                processed.append('</details>\n\n')
            
            title = line.strip().lstrip('#').strip()
            processed.append(f'<details>\n<summary><b style="font-size:1.2em">{title}</b></summary>\n<br>\n\n')
            in_details = True
        elif line.startswith('### '):
            title = line.strip().lstrip('#').strip()
            processed.append(f'<br>\n\n**{title}**\n')
        elif line.startswith('## '):
            title = line.strip().lstrip('#').strip()
            processed.append(f'<br>\n\n**{title}**\n')
        else:
            # Handle image placeholders like \[ \text{Hình 3-1: ...} \]
            match = re.search(r'\\\\\[\\text\{(Hình 3-(\d+).*?)\}\\\\\]', line)
            if match:
                full_text = match.group(1)
                img_num = match.group(2)
                replacement = f'![{full_text}](../Figures/CH03/Hinh_3-{img_num}.png)\n*{full_text}*\n'
                line = line[:match.start()] + replacement + line[match.end():]
            
            # Handle the cases where it says \[ \text{[some_digit_plot...]} \]
            match2 = re.search(r'\\\\\[\\text\{\[(.*?)\]\}\\\\\]', line)
            if match2:
                line = line[:match2.start()] + f'*{match2.group(1)}*' + line[match2.end():]
                
            # If line is '---', we can just keep it or remove it. Let's keep it.
            processed.append(line)
            
    if in_details:
        processed.append('</details>\n')

    final_glossary_content = "".join(processed)

    # 4. Inject into chuong_03.md
    with open('docs/chuong_03.md', 'r', encoding='utf-8') as f:
        chuong03_content = f.read()

    # Find the start and end of the current Glossary Tab
    start_marker = "<!-- tabs:start -->"
    end_marker = "#### ** 📖 Lý thuyết **"
    
    start_idx = chuong03_content.find(start_marker)
    end_idx = chuong03_content.find(end_marker)
    
    if start_idx != -1 and end_idx != -1:
        new_tab_content = f"""<!-- tabs:start -->

#### ** 📚 Thuật ngữ & Khái niệm **

*Dưới đây là tổng hợp toàn bộ các thuật ngữ, khái niệm cốt lõi, công thức và mã nguồn minh họa trong Chương 3 để bạn tra cứu nhanh.*

{final_glossary_content}

<br/>

"""
        new_chuong03 = chuong03_content[:start_idx] + new_tab_content + chuong03_content[end_idx:]
        
        with open('docs/chuong_03.md', 'w', encoding='utf-8') as f:
            f.write(new_chuong03)
        print("Updated chuong_03.md successfully.")
    else:
        print("Markers not found in chuong_03.md")

if __name__ == '__main__':
    process_glossary()
