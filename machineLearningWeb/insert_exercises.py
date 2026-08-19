import os
import re

TEXT_DIR = r"d:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Máy học_2026\machineLearningWeb\TaiLieu\Excercise"
DOCS_DIR = r"d:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Máy học_2026\machineLearningWeb\docs"

def parse_exercise_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    intro_text = []
    questions = []
    
    current_q = None
    state = 'intro' 
    
    for line in lines:
        if re.match(r'^(#+)\s*\*\*Câu \d+.*', line, re.IGNORECASE) or re.match(r'^\d+\.\s*\*\*Câu \d+.*', line, re.IGNORECASE):
            # Skip the lines that are just in the Table of Contents, which start with "1. **Câu 1**: " etc.
            # Usually detailed questions are under `#` tags like `###### **Câu 1:...`
            # But wait, what if the user used `1. **Câu 1**:` for the actual question? Let's assume actual questions have newlines and follow the structure.
            # The TOC lines usually don't have Lời giải chi tiết after them. Let's just treat all as potential, but if it has no solution_lines it will be empty anyway.
            # Actually, `1. **Câu 1**` in the TOC has a different format. Let's just stick to the `#` heading format for actual questions.
            pass
            
        if re.match(r'^(#+)\s*\*\*Câu \d+.*', line, re.IGNORECASE):
            if current_q:
                questions.append(current_q)
            # Clean title
            title = line.strip()
            # Remove leading '#'
            title = re.sub(r'^#+\s*', '', title)
            # Remove leading/trailing '**'
            title = title.replace('**', '').strip()
            
            current_q = {
                'title': title,
                'question_lines': [],
                'solution_lines': []
            }
            state = 'question'
        elif current_q and re.match(r'^\*\s+\*\*Lời giải.*', line, re.IGNORECASE):
            state = 'solution'
            current_q['solution_lines'].append(line)
        else:
            if state == 'intro':
                intro_text.append(line)
            elif state == 'question':
                if re.match(r'^\*\s+\*\*Yêu cầu đề bài.*', line, re.IGNORECASE):
                    continue
                current_q['question_lines'].append(line)
            elif state == 'solution':
                current_q['solution_lines'].append(line)
                
    if current_q:
        questions.append(current_q)
        
    return "".join(intro_text), questions

def sanitize_markdown(text):
    # Convert #### **Text** to ##### **Text** to avoid docsify-tabs thinking it's a new tab
    return re.sub(r'^#### \*\*(.*?)\*\*', r'##### **\1**', text, flags=re.MULTILINE)

def format_html(intro_text, questions):
    html = ["\n#### ** 📝 Bài Tập **\n\n"]
    html.append("""
<script>
if (typeof checkPasswordAndShow !== 'function') {
  window.checkPasswordAndShow = function(btn) {
    let password = prompt("Vui lòng nhập mật khẩu để xem lời giải:");
    if (password === "donga2026") {
      let content = btn.nextElementSibling;
      if (content && content.classList.contains("solution-content")) {
        content.style.display = "block";
        btn.style.display = "none";
      }
    } else {
      alert("Mật khẩu không đúng!");
    }
  };
}
</script>
""")
    html.append(sanitize_markdown(intro_text) + "\n")

    
    for q in questions:
        title = q['title']
        question_html = sanitize_markdown("".join(q['question_lines']).strip())
        solution_html = sanitize_markdown("".join(q['solution_lines']).strip())
        
        box = f"""
<div class="exercise-box" style="background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
  <h4 style="color: #1a73e8; margin-top: 0;">{title}</h4>
  
{question_html}

  <details style="margin-top: 15px; margin-bottom: 15px; background: #f8faff; padding: 10px; border-radius: 6px; border-left: 4px solid #1a73e8;">
    <summary style="font-weight: bold; cursor: pointer; color: #1a73e8;">💡 Gợi ý</summary>
    <div style="margin-top: 10px;">
      Hãy phân tích kỹ các khái niệm trong bài học và áp dụng vào yêu cầu của đề bài. Đọc lại phần lý thuyết liên quan nếu cần.
    </div>
  </details>
  
  <div class="solution-section">
    <button onclick="checkPasswordAndShow(this)" style="background: #34a853; color: white; border: none; padding: 8px 16px; border-radius: 20px; font-weight: bold; cursor: pointer; transition: background 0.3s;">🔑 Xem lời giải</button>
    <div class="solution-content" style="display: none; margin-top: 15px; padding-top: 15px; border-top: 1px dashed #ccc;">

{solution_html}

    </div>
  </div>
</div>
"""
        html.append(box)
    
    return "".join(html)

def update_chapter(chap_num):
    txt_file = os.path.join(TEXT_DIR, f"ExcerciseChapter_{chap_num}.txt")
    md_file = os.path.join(DOCS_DIR, f"chuong_{chap_num}.md")
    
    if not os.path.exists(txt_file):
        print(f"Skipping chap {chap_num}: {txt_file} not found.")
        return
        
    if not os.path.exists(md_file):
        print(f"Skipping chap {chap_num}: {md_file} not found.")
        return
        
    intro_text, questions = parse_exercise_file(txt_file)
    exercises_html = format_html(intro_text, questions)
    
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
    start_idx = content.find("#### ** 📝 Bài Tập **")
    if start_idx == -1:
        start_idx = content.find("#### ** 📝 Bài tập **")
        
    end_idx = content.find("<!-- tabs:end -->")
    
    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
        new_content = content[:start_idx] + exercises_html + "\n\n" + content[end_idx:]
    elif end_idx != -1:
        new_content = content[:end_idx] + exercises_html + "\n\n" + content[end_idx:]
    else:
        new_content = content + "\n\n" + exercises_html
        
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(new_content)
        
    print(f"Updated chuong_{chap_num}.md with {len(questions)} questions.")

if __name__ == "__main__":
    for i in range(10, 20):
        update_chapter(i)
