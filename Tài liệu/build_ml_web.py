import os
import re
import shutil
from bs4 import BeautifulSoup

html_file = r"d:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Máy học_2026\Tài liệu\scratch\extracted.html"
images_src_dir = r"d:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Máy học_2026\Tài liệu\Hands-On Machine Learning with ScikitLearn, Keras, and TensorFlow2_files"
out_docs_dir = r"d:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Máy học_2026\machineLearningWeb\docs"
out_figures_dir = r"d:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Máy học_2026\machineLearningWeb\Figures"

os.makedirs(out_docs_dir, exist_ok=True)
os.makedirs(out_figures_dir, exist_ok=True)

print("Parsing HTML...")
with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
    soup = BeautifulSoup(f, 'html.parser')

print("Extracting content...")
chapters = []
current_chapter = None
current_content = []
current_chapter_idx = 0

last_text = ""
last_image_src = None
sidebar_links = []

# Process all tags in document order
# Word HTML often wraps paragraphs in <p class="MsoNormal">
# Images are in <v:imagedata> or <img>
for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'table', 'v:imagedata', 'img']):
    if tag.name != 'table' and tag.find_parent('table'):
        continue

    if tag.name in ['v:imagedata', 'img']:
        src = tag.get('src')
        if src:
            # src is like "Hands-OnMachineLearningwithScikitLearn,Keras,andTensorFlow_files/image001.jpg"
            filename = os.path.basename(src)
            # The actual directory is "Hands-On Machine Learning with ScikitLearn, Keras, and TensorFlow2_files"
            # It seems the MHT used a slightly different folder name in the src path.
            # But we just need the filename.
            last_image_src = filename
            continue

    text = tag.text.strip()
    if not text:
        continue
        
    # De-duplicate (Word puts same text multiple times)
    if text == last_text:
        continue
    last_text = text

    # Check for Headings
    if tag.name == 'h1':
        current_content.append(f"\n# {text}\n")
    elif tag.name == 'h2':
        # Check if it's a new chapter
        if re.match(r'^(CHƯƠNG|PHỤ LỤC)', text, re.IGNORECASE):
            # Save previous chapter
            if current_chapter:
                chapters.append((current_chapter, current_content))
            
            current_chapter = text.replace(':', ' -')
            current_chapter_idx += 1
            current_content = [f"\n# {text}\n"]
            
            # Add to sidebar
            ch_filename = f"docs/chuong_{current_chapter_idx:02d}.md"
            sidebar_links.append(f"* [{text}]({ch_filename})")
        else:
            current_content.append(f"\n## {text}\n")
    elif tag.name == 'h3':
        current_content.append(f"\n### {text}\n")
    elif tag.name == 'h4':
        current_content.append(f"\n#### {text}\n")
    elif tag.name == 'p':
        # Check if it's a caption
        caption_match = re.match(r'^(Hình|Figure)\s+(\d+-\d+)', text, re.IGNORECASE)
        if caption_match:
            img_id = caption_match.group(2) # e.g. "1-1"
            if last_image_src:
                # We need to save the image
                ch_dir = f"CH{current_chapter_idx:02d}"
                ch_fig_dir = os.path.join(out_figures_dir, ch_dir)
                os.makedirs(ch_fig_dir, exist_ok=True)
                
                # Find the extension from last_image_src
                ext = '.jpg'
                if last_image_src.lower().endswith('.png') or last_image_src.lower().endswith('.pn_'): ext = '.png'
                
                src_base = last_image_src.split('.')[0] # e.g. image001
                src_file_found = None
                for ext_guess in ['.jp_', '.pn_', '.jpg', '.png']:
                    guess_path = os.path.join(images_src_dir, src_base + ext_guess)
                    if os.path.exists(guess_path):
                        src_file_found = guess_path
                        break
                
                new_img_name = f"Hinh_{img_id}{ext}"
                dest_path = os.path.join(ch_fig_dir, new_img_name)
                
                if src_file_found:
                    shutil.copy2(src_file_found, dest_path)
                
                if not src_file_found and not os.path.exists(dest_path):
                    alt_ext = '.png' if ext == '.jpg' else '.jpg'
                    alt_dest = os.path.join(ch_fig_dir, f"Hinh_{img_id}{alt_ext}")
                    if os.path.exists(alt_dest):
                        new_img_name = f"Hinh_{img_id}{alt_ext}"
                    else:
                        print(f"Warning: Could not find source or existing image for {last_image_src}")

                # Add image to markdown
                current_content.append(f"\n![{text}](../Figures/{ch_dir}/{new_img_name})\n")
                    
                last_image_src = None # Reset
            
            # Add caption text as bold or italic
            current_content.append(f"\n*{text}*\n")
        else:
            classes = tag.get('class', [])
            if isinstance(classes, str): classes = [classes]
            
            if 'SourceCode' in classes:
                current_content.append(f"\n```python\n{text}\n```\n")
            else:
                current_content.append(f"{text}\n\n")
    elif tag.name == 'table':
        table_md = []
        rows = tag.find_all('tr')
        if rows:
            for i, tr in enumerate(rows):
                cols = tr.find_all(['th', 'td'])
                col_texts = [c.text.strip().replace('\n', '<br>') for c in cols]
                table_md.append("| " + " | ".join(col_texts) + " |")
                if i == 0:
                    table_md.append("|" + "|".join(['---'] * len(col_texts)) + "|")
            
            current_content.append("\n" + "\n".join(table_md) + "\n\n")

# Add the last chapter
if current_chapter:
    chapters.append((current_chapter, current_content))

print(f"Found {len(chapters)} chapters.")

# Write chapters to markdown files
for i, (ch_title, content) in enumerate(chapters):
    filename = f"chuong_{i+1:02d}.md"
    filepath = os.path.join(out_docs_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("\n".join(content))
    print(f"Wrote {filename}")

# Write sidebar
sidebar_path = os.path.join(out_docs_dir, '_sidebar.md')
with open(sidebar_path, 'w', encoding='utf-8') as f:
    f.write("* [Trang chủ](/)\n\n")
    f.write("* **Mục lục**\n")
    for link in sidebar_links:
        f.write(f"  {link}\n")

print("Done building course web.")
