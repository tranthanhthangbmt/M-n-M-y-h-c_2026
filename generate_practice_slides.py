import json
import re
import os
import base64

try:
    from deep_translator import GoogleTranslator
    translator = GoogleTranslator(source='en', target='vi')
except ImportError:
    translator = None

def translate_text(text):
    if not translator: return text
    if not text.strip(): return text
    
    # Protect inline code blocks
    code_blocks = []
    def code_repl(match):
        code_blocks.append(match.group(0))
        return f" __CODE_{len(code_blocks)-1}__ "
        
    text_prot = re.sub(r'`[^`]+`', code_repl, text)
    
    try:
        translated = translator.translate(text_prot)
        if not translated:
            translated = text_prot
    except Exception:
        translated = text_prot
        
    # Restore code blocks
    for i, code in enumerate(code_blocks):
        # translator might alter the placeholder format (e.g., lowercased or added spaces)
        translated = re.sub(rf"(?i)__\s*CODE\s*_{i}\s*__", code, translated)
        
    return translated

ipynb_path = r"D:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Máy học_2026\machineLearningWeb\TaiLieu\NotebookJupyter\01_the_machine_learning_landscape.ipynb"
tex_path = r"D:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Máy học_2026\slideML\Slide_ML_Chap01_Practice_01_the_machine_learning_landscape.tex"
images_dir = r"D:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Máy học_2026\slideML\images\ch01"

if not os.path.exists(images_dir):
    os.makedirs(images_dir)

with open(ipynb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def clean_markdown(text):
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    text = re.sub(r'<.*?>', '', text)
    return text

def escape_tex(text):
    text = clean_markdown(text)
    text = text.replace('\\', '\\textbackslash ')
    text = text.replace('%', '\\%')
    text = text.replace('$', '\\$')
    text = text.replace('#', '\\#')
    text = text.replace('_', '\\_')
    text = text.replace('&', '\\&')
    text = text.replace('^', '\\textasciicircum ')
    text = text.replace('~', '\\textasciitilde ')
    text = text.replace('**', '')
    text = text.replace('`', '')
    return text

def split_code(code, max_lines=15):
    lines = code.split('\n')
    return ['\n'.join(lines[i:i+max_lines]) for i in range(0, len(lines), max_lines)]

tex_header = r"""\documentclass[aspectratio=169]{beamer}
\usetheme{Madrid}
\usecolortheme{default}
\setbeamertemplate{caption}{\raggedright\insertcaption\par}
\usepackage{fontspec}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{listings}

\lstset{
    language=Python,
    basicstyle=\ttfamily\small,
    keywordstyle=\color{blue},
    stringstyle=\color{red},
    commentstyle=\color{green!50!black},
    showstringspaces=false,
    breaklines=true,
    frame=single,
    backgroundcolor=\color{gray!10}
}

\title[Thực hành Chương 1]{THỰC HÀNH: TỔNG QUAN HỌC MÁY \\ \vspace{0.5cm} \Large Hướng dẫn chi tiết}
\author{TS. Trần Thành Thắng}
\institute{Đại học Đông Á}
\date{\today}

\begin{document}
"""

tex_footer = r"""
\end{document}
"""

frames = []
current_frame_content = ""
current_frame_lines = 0
section_title = "Hướng dẫn thực hành"
frame_part_number = 1

def flush_frame():
    global current_frame_content, current_frame_lines, section_title, frame_part_number, frames
    if current_frame_content.strip():
        fragile = "\\begin{lstlisting}" in current_frame_content or "\\begin{verbatim}" in current_frame_content
        options = "[fragile]" if fragile else "[allowframebreaks]"
        
        display_title = section_title
        if frame_part_number > 1 and not options == "[allowframebreaks]":
            display_title += f" (Tiếp theo {frame_part_number-1})"
            
        frames.append(r"""
\begin{frame}%s{%s}
%s
\end{frame}
""" % (options, display_title, current_frame_content))
        current_frame_content = ""
        current_frame_lines = 0
        frame_part_number += 1

def add_to_frame(content, lines_needed, title):
    global current_frame_content, current_frame_lines, section_title, frame_part_number
    if title != section_title:
        if current_frame_lines > 0:
            flush_frame()
        section_title = title
        frame_part_number = 1
    elif current_frame_lines + lines_needed > 14 and current_frame_lines > 0:
        flush_frame()
        
    current_frame_content += content + "\n\n"
    current_frame_lines += lines_needed

for i, cell in enumerate(nb['cells']):
    ctype = cell['cell_type']
    source = "".join(cell.get('source', []))
    
    if not source.strip():
        continue
        
    if ctype == 'markdown':
        m = re.match(r'^#+\s+(.*)', source.split('\n')[0])
        if m:
            title_text = m.group(1)
            translated_title = translate_text(title_text)
            title = escape_tex(translated_title)
        else:
            title = section_title
            
        translated_source = translate_text(source)
        content = escape_tex(translated_source)
        
        # ensure newlines in markdown become paragraph breaks in LaTeX
        content = content.replace('\n', '\n\n')
        
        # Estimate lines
        lines_needed = sum(max(1, len(p)//80 + 1) for p in content.split('\n\n'))
        
        add_to_frame(content, lines_needed, title)
        
    elif ctype == 'code':
        # Split code into multiple chunks if too long
        chunks = split_code(source.strip(), max_lines=14)
        for idx, chunk in enumerate(chunks):
            code_content = f"\\begin{{lstlisting}}[language=Python]\n{chunk}\n\\end{{lstlisting}}"
            lines_needed = len(chunk.split('\n')) + 2
            add_to_frame(code_content, lines_needed, section_title)
        
        # Outputs
        outputs = cell.get('outputs', [])
        for out_idx, out in enumerate(outputs):
            if out.get('output_type') == 'stream':
                text = "".join(out.get('text', []))
                lines = text.strip().split('\n')
                if len(lines) > 15:
                    text = '\n'.join(lines[:15]) + '\n... (đã cắt bớt)'
                    lines_needed = 16
                else:
                    text = '\n'.join(lines)
                    lines_needed = len(lines) + 2
                    
                out_content = f"\\textbf{{Kết quả:}}\n\\begin{{verbatim}}\n{text}\n\\end{{verbatim}}"
                add_to_frame(out_content, lines_needed + 1, section_title)
                
            elif out.get('output_type') in ('display_data', 'execute_result'):
                if 'image/png' in out.get('data', {}):
                    img_data = out['data']['image/png']
                    img_bytes = base64.b64decode(img_data)
                    img_filename = f"cell_{i}_out_{out_idx}.png"
                    img_path = os.path.join(images_dir, img_filename)
                    with open(img_path, "wb") as f_img:
                        f_img.write(img_bytes)
                    
                    img_content = r"""\begin{center}
\includegraphics[width=\textwidth,height=0.75\textheight,keepaspectratio]{images/ch01/%s}
\end{center}""" % img_filename
                    
                    # Force flush before image, add image as full slide, then flush after image
                    if current_frame_lines > 0: flush_frame()
                    add_to_frame(img_content, 20, section_title + " (Biểu đồ)")
                    flush_frame()
                    
                elif 'text/plain' in out.get('data', {}):
                    text = "".join(out['data']['text/plain'])
                    lines = text.strip().split('\n')
                    if len(lines) > 15:
                        text = '\n'.join(lines[:15]) + '\n... (đã cắt bớt)'
                        lines_needed = 16
                    else:
                        text = '\n'.join(lines)
                        lines_needed = len(lines) + 2
                        
                    out_content = f"\\textbf{{Kết quả:}}\n\\begin{{verbatim}}\n{text}\n\\end{{verbatim}}"
                    add_to_frame(out_content, lines_needed + 1, section_title)

# flush the last frame
flush_frame()

with open(tex_path, 'w', encoding='utf-8') as f:
    f.write(tex_header)
    for frame in frames:
        f.write(frame)
    f.write(tex_footer)

print(f"Generated {tex_path}")
