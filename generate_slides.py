import os
import re

md_path = r"d:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Máy học_2026\machineLearningWeb\docs\chuong_01.md"
tex_path = r"d:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Máy học_2026\slideML\Slide_ML_Chap01.tex"

with open(md_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

frames = []
current_frame_title = "Giới thiệu"
current_content = []
MAX_ITEMS_PER_SLIDE = 4

def flush_frame():
    global current_content, frames
    if current_content:
        frames.append({
            'title': current_frame_title,
            'content': current_content
        })
        current_content = []

in_code_block = False
code_content = []
in_table = False
table_content = []

def escape_tex(text):
    # Split by $...$ to preserve math
    parts = re.split(r'(\$.*?\$)', text)
    res = []
    for p in parts:
        if p.startswith('$') and p.endswith('$'):
            res.append(p)
        else:
            # Escape LaTeX special chars
            p = p.replace('\\', '\\\\')
            p = p.replace('%', r'\%')
            p = p.replace('&', r'\&')
            p = p.replace('#', r'\#')
            p = p.replace('_', r'\_')
            p = p.replace('^', r'\^')
            p = p.replace('~', r'\~')
            p = p.replace('θ', r'$\theta$')
            p = p.replace('α', r'$\alpha$')
            p = p.replace('β', r'$\beta$')
            p = p.replace('×', r'$\times$')
            p = p.replace('–', '-') # en-dash to hyphen
            res.append(p)
    return "".join(res)

for line in lines:
    line = line.strip()
    if not line:
        continue
        
    if line.startswith('```'):
        if not in_code_block:
            in_code_block = True
            lang = line[3:].strip()
            if not lang: lang = 'Python'
            code_content = []
        else:
            in_code_block = False
            flush_frame()
            frames.append({
                'title': current_frame_title + " (Code)",
                'content': ['CODE', lang, "\n".join(code_content)],
                'fragile': True
            })
        continue
        
    if in_code_block:
        code_content.append(line)
        continue
        
    if line.startswith('#'):
        m = re.match(r'^#+\s+(.*)', line)
        if m:
            flush_frame()
            title = m.group(1)
            title = escape_tex(title)
            if line.startswith('# '):
                frames.append({'section': title})
            elif line.startswith('## '):
                frames.append({'subsection': title})
            current_frame_title = title
        continue
        
    if line.startswith('![') or line.startswith('<img') or '<v:imagedata' in line:
        m = re.search(r'!\[(.*?)\]\((.*?)\)', line)
        if m:
            caption = escape_tex(m.group(1))
            img_path = m.group(2)
            # Make path relative to slideML folder
            img_path = img_path.replace('../', '../machineLearningWeb/')
            flush_frame()
            frames.append({
                'title': current_frame_title + " (Hình ảnh)",
                'content': ['IMAGE', img_path, caption]
            })
        continue
        
    if line.startswith('|'):
        if not in_table:
            in_table = True
            flush_frame()
            table_content = []
        table_content.append(line)
        continue
    else:
        if in_table:
            in_table = False
            frames.append({
                'title': current_frame_title + " (Bảng)",
                'content': ['TABLE', "\n".join(table_content)]
            })
            table_content = []
            
    if line.startswith('>'):
        line = line[1:].strip()
            
    # Regular text
    # Let's split into sentences if it's too long, but for now just add as a bullet point.
    line = escape_tex(line)
    
    # If line is extremely long, we might split it by '.' but simple bullet points are okay.
    current_content.append(line)
    if len(current_content) >= MAX_ITEMS_PER_SLIDE:
        flush_frame()
        if not current_frame_title.endswith("(Tiếp)"):
            current_frame_title = current_frame_title + " (Tiếp)"
        
flush_frame()
if in_table:
    frames.append({
        'title': current_frame_title + " (Bảng)",
        'content': ['TABLE', "\n".join(table_content)]
    })

print(f"Total frames extracted: {len(frames)}")

with open(tex_path, 'w', encoding='utf-8') as f:
    f.write(r'''\documentclass[aspectratio=169]{beamer}
\usetheme{Madrid}
\usecolortheme{default}
\usepackage{fontspec}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{amsmath}

\definecolor{codegreen}{rgb}{0,0.6,0}
\definecolor{codegray}{rgb}{0.5,0.5,0.5}
\definecolor{codepurple}{rgb}{0.58,0,0.82}
\definecolor{backcolour}{rgb}{0.95,0.95,0.92}

\lstdefinestyle{mystyle}{
    backgroundcolor=\color{backcolour},   
    commentstyle=\color{codegreen},
    keywordstyle=\color{magenta},
    numberstyle=\tiny\color{codegray},
    stringstyle=\color{codepurple},
    basicstyle=\ttfamily\footnotesize,
    breakatwhitespace=false,         
    breaklines=true,                 
    captionpos=b,                    
    keepspaces=true,                 
    numbers=left,                    
    numbersep=5pt,                  
    showspaces=false,                
    showstringspaces=false,
    showtabs=false,                  
    tabsize=2
}
\lstset{style=mystyle}

\setbeamertemplate{caption}[numbered]
\renewcommand{\figurename}{Hình}
\renewcommand{\tablename}{Bảng}

\title[Chương 1: Tổng quan Học máy]{Học máy \\ \vspace{0.3cm} \Large Chương 1: Bức tranh tổng quan về Học máy}
\author{Giảng viên: TS. Trần Thành Thắng}
\date{\today}

\begin{document}

\begin{frame}
    \titlepage
\end{frame}

\begin{frame}{Nội dung}
    \tableofcontents[hideallsubsections]
\end{frame}
''')

    for frame in frames:
        if 'section' in frame:
            f.write(f"\\section{{{frame['section']}}}\n")
            f.write(f"\\begin{{frame}}\n\\vfill\\centering\\LARGE\\textbf{{{frame['section']}}}\n\\vfill\\end{{frame}}\n\n")
            continue
        if 'subsection' in frame:
            f.write(f"\\subsection{{{frame['subsection']}}}\n")
            continue
            
        is_fragile = frame.get('fragile', False)
        frag_str = "[fragile]" if is_fragile else ""
        title = frame['title']
        f.write(f"\\begin{{frame}}{frag_str}{{{title}}}\n")
        
        content = frame['content']
        if content and content[0] == 'CODE':
            lang = content[1]
            code = content[2]
            f.write(f"\\begin{{lstlisting}}[language={lang}]\n{code}\n\\end{{lstlisting}}\n")
        elif content and content[0] == 'IMAGE':
            img_path = content[1]
            caption = content[2]
            f.write(f"\\begin{{figure}}[ht]\n\\centering\n\\includegraphics[height=0.65\\textheight, keepaspectratio]{{{img_path}}}\n\\caption{{{caption}}}\n\\end{{figure}}\n")
        elif content and content[0] == 'TABLE':
            tbl_lines = content[1].split('\n')
            f.write("\\begin{table}\n\\centering\n\\small\n")
            
            header = tbl_lines[0]
            cols = [c.strip() for c in header.split('|') if c.strip()]
            num_cols = len(cols)
            f.write("\\begin{tabular}{" + "c " * num_cols + "}\n\\toprule\n")
            for i, row in enumerate(tbl_lines):
                if '---' in row:
                    f.write("\\midrule\n")
                    continue
                cells = [c.strip() for c in row.split('|') if c.strip()]
                if not cells: continue
                while len(cells) < num_cols: cells.append("")
                cells = [escape_tex(c) for c in cells]
                # tables often contain <br>. Replace with space or newline
                cells = [c.replace('<br>', ' ') for c in cells]
                f.write(" & ".join(cells) + " \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
        else:
            f.write("\\begin{itemize}\n")
            for item in content:
                # If an item is just an empty string somehow, skip it
                if item.strip():
                    f.write(f"\\item {item}\n")
            f.write("\\end{itemize}\n")
            
        f.write("\\end{frame}\n\n")

    f.write(r'\end{document}')

print("Done generating TeX file.")
