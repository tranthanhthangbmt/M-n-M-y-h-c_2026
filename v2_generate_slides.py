import os
import re

md_path = r"d:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Máy học_2026\machineLearningWeb\docs\chuong_01.md"
tex_path = r"d:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Máy học_2026\slideML\Slide_ML_Chap01.tex"

def escape_tex(text):
    if not text: return text
    # handle markdown bold/italic before escaping
    text = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', text)
    text = re.sub(r'\*(.*?)\*', r'\\textit{\1}', text)
    
    # escape special tex characters outside of math
    # this is a simple escaper
    parts = re.split(r'(\$.*?\$|\\\[.*?\\\])', text)
    for i in range(0, len(parts), 2):
        p = parts[i]
        if p:
            p = p.replace('\\', '\\textbackslash ')
            p = p.replace('%', r'\%')
            p = p.replace('&', r'\&')
            p = p.replace('#', r'\#')
            p = p.replace('_', r'\_')
            p = p.replace('{', r'\{')
            p = p.replace('}', r'\}')
            p = p.replace('~', r'\textasciitilde ')
            p = p.replace('^', r'\textasciicircum ')
            
            # handle some unicode chars
            p = p.replace('—', '---')
            p = p.replace('–', '--')
            p = p.replace('“', '``')
            p = p.replace('”', "''")
            p = p.replace('‘', '`')
            p = p.replace('’', "'")
            p = p.replace('α', r'$\alpha$')
            p = p.replace('β', r'$\beta$')
            p = p.replace('γ', r'$\gamma$')
            p = p.replace('θ', r'$\theta$')
            p = p.replace('∈', r'$\in$')
            p = p.replace('−', r'$-$')
            
            parts[i] = p
    return "".join(parts)

def parse_markdown(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    paragraphs = content.split('\n\n')
    frames = []
    
    # Add learning objectives
    frames.append({
        'type': 'objectives',
        'content': []
    })

    current_section = None
    current_subsection = None
    
    in_code_block = False
    code_content = []
    code_lang = ""

    for para in paragraphs:
        para = para.strip()
        if not para: continue
        
        if para.startswith('```'):
            if not in_code_block:
                in_code_block = True
                code_lang = para.strip('`').strip()
                code_content = []
            else:
                in_code_block = False
                frames.append({
                    'type': 'code',
                    'lang': code_lang,
                    'content': '\n'.join(code_content),
                    'section': current_section,
                    'subsection': current_subsection
                })
                code_content = []
            continue
            
        if in_code_block:
            code_content.append(para)
            continue
            
        # Check headings
        if para.startswith('# '):
            current_section = para[2:].strip()
            frames.append({'type': 'section', 'title': current_section})
            continue
        elif para.startswith('## ') or para.startswith('### '):
            current_subsection = para.lstrip('#').strip()
            frames.append({'type': 'subsection', 'title': current_subsection})
            continue
            
        # Check images
        img_match = re.match(r'!\[(.*?)\]\((.*?)\)', para)
        if img_match:
            frames.append({
                'type': 'image',
                'caption': img_match.group(1),
                'path': img_match.group(2),
                'section': current_section,
                'subsection': current_subsection
            })
            continue
            
        # Text paragraph
        # Merge newlines inside paragraph
        para = para.replace('\n', ' ')
        # split into sentences roughly by dot
        sentences = re.split(r'(?<=[.!?]) +', para)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if sentences:
            # chunk sentences into groups of 3
            for i in range(0, len(sentences), 3):
                chunk = sentences[i:i+3]
                frames.append({
                    'type': 'text',
                    'content': chunk,
                    'section': current_section,
                    'subsection': current_subsection
                })

    return frames

def generate_latex(frames, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(r"""\documentclass[aspectratio=169]{beamer}
\usetheme{Madrid}
\usecolortheme{default}
\usepackage{fontspec}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{listings}
\usepackage{xcolor}

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

\title{CHƯƠNG 1. BỨC TRANH TỔNG QUAN VỀ HỌC MÁY}
\author{TS. Trần Thành Thắng}
\institute{Đại học Đông Á}
\date{\today}

\begin{document}

\begin{frame}
\titlepage
\end{frame}

\begin{frame}{Nội dung}
\tableofcontents[hideallsubsections]
\end{frame}
""")

        for frame in frames:
            if frame['type'] == 'objectives':
                f.write(r"""\begin{frame}{Mục tiêu bài học}
\begin{itemize}
    \item Hiểu rõ định nghĩa và tầm quan trọng của Học máy (Machine Learning).
    \item Phân biệt các hệ thống Học máy (Có giám sát, Không giám sát, Bán giám sát, Học tăng cường).
    \item Nắm bắt quy trình phát triển một dự án Học máy thực tế.
    \item Nhận diện các thách thức chính (Overfitting, Underfitting, Chất lượng dữ liệu).
\end{itemize}
\end{frame}
""")
            elif frame['type'] == 'section':
                f.write(f"\\section{{{escape_tex(frame['title'] )}}}\n")
                f.write(f"\\begin{{frame}}\n\\vfill\\centering\\LARGE\\textbf{{{escape_tex(frame['title'] )}}}\n\\vfill\\end{{frame}}\n\n")
            elif frame['type'] == 'subsection':
                f.write(f"\\subsection{{{escape_tex(frame['title'] )}}}\n")
            elif frame['type'] == 'image':
                title = escape_tex(frame['subsection'] or frame['section'] or "Hình ảnh")
                caption = escape_tex(frame['caption'])
                # adjust path
                img_path = frame['path']
                if img_path.startswith('/'):
                    img_path = r"d:/DongAUniversity/TÀI LIỆU DẠY HỌC_2024-2025/Môn Máy học_2026/machineLearningWeb/docs" + img_path
                else:
                    img_path = r"d:/DongAUniversity/TÀI LIỆU DẠY HỌC_2024-2025/Môn Máy học_2026/machineLearningWeb/docs/" + img_path

                
                f.write(f"\\begin{{frame}}{{{title}}}\n")
                f.write(f"\\begin{{figure}}\n")
                f.write(f"\\centering\n")
                f.write(f"\\includegraphics[width=0.8\\textwidth,height=0.75\\textheight,keepaspectratio]{{{img_path}}}\n")
                f.write(f"\\caption{{{caption}}}\n")
                f.write(f"\\end{{figure}}\n")
                f.write(f"\\end{{frame}}\n\n")
            elif frame['type'] == 'code':
                title = escape_tex(frame['subsection'] or frame['section'] or "Mã nguồn")
                f.write(f"\\begin{{frame}}[fragile]{{{title}}}\n")
                lang = frame['lang'] if frame['lang'] else 'Python'
                f.write(f"\\begin{{lstlisting}}[language={lang}]\n")
                f.write(frame['content'] + "\n")
                f.write(f"\\end{{lstlisting}}\n")
                f.write(f"\\end{{frame}}\n\n")
            elif frame['type'] == 'text':
                title = escape_tex(frame['subsection'] or frame['section'] or "Nội dung")
                f.write(f"\\begin{{frame}}{{{title}}}\n")
                
                # Check if it contains "Định nghĩa" or similar
                joined = " ".join(frame['content'])
                if "Học máy là" in joined or "Định nghĩa" in joined:
                    f.write(f"\\begin{{block}}{{Khái niệm quan trọng}}\n")
                    f.write("\\begin{itemize}\n")
                    for s in frame['content']:
                        f.write(f"  \\item {escape_tex(s)}\n")
                    f.write("\\end{itemize}\n")
                    f.write(f"\\end{{block}}\n")
                elif "Ví dụ" in joined:
                    f.write(f"\\begin{{exampleblock}}{{Ví dụ}}\n")
                    f.write("\\begin{itemize}\n")
                    for s in frame['content']:
                        f.write(f"  \\item {escape_tex(s)}\n")
                    f.write("\\end{itemize}\n")
                    f.write(f"\\end{{exampleblock}}\n")
                else:
                    f.write("\\begin{itemize}\n")
                    for s in frame['content']:
                        f.write(f"  \\item {escape_tex(s)}\n")
                    f.write("\\end{itemize}\n")
                
                f.write(f"\\end{{frame}}\n\n")

        f.write(r"\end{document}")

if __name__ == "__main__":
    frames = parse_markdown(md_path)
    generate_latex(frames, tex_path)
    print("LaTeX files generated successfully!")
