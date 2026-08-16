import os
import re

# Configuration
NOTEBOOK_DIR = r"D:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Máy học_2026\machineLearningWeb\TaiLieu\NotebookJupyter"
DOCS_DIR = r"D:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Máy học_2026\machineLearningWeb\docs"
GITHUB_BASE_URL = "https://colab.research.google.com/github/tranthanhthangbmt/M-n-M-y-h-c_2026/blob/main/machineLearningWeb/TaiLieu/NotebookJupyter/"
LOCAL_BASE_URL = "TaiLieu/NotebookJupyter/"

# Data structure to hold parsed notebooks
# chapters = { "01": { "1_the_machine_learning_landscape": { 'order': 1, 'name': ..., 'display_name': ..., 'en_file': ..., 'vn_file': ...} } }
chapters = {}

def parse_notebooks():
    files = [f for f in os.listdir(NOTEBOOK_DIR) if f.endswith('.ipynb') and not f.endswith('.pdf')]
    for f in files:
        # Regex to match formats like:
        # 01.1_the_machine_learning_landscape_VN.ipynb
        # 03_classification_VN.ipynb
        # 10.2_extra_ann_architectures_EN.ipynb
        m = re.match(r'^(\d+)(?:\.(\d+))?_(.+?)_(EN|VN)\.ipynb$', f)
        if not m:
            continue
            
        chap_num = m.group(1)
        order = int(m.group(2)) if m.group(2) else 1
        name = m.group(3)
        lang = m.group(4)
        
        # Clean up name for display
        display_name = name.replace('_', ' ').title()
        
        if chap_num not in chapters:
            chapters[chap_num] = {}
            
        key = f"{order}_{name}"
        if key not in chapters[chap_num]:
            chapters[chap_num][key] = {
                'order': order,
                'name': name,
                'display_name': display_name,
                'en_file': None,
                'vn_file': None
            }
            
        if lang == 'EN':
            chapters[chap_num][key]['en_file'] = f
        else:
            chapters[chap_num][key]['vn_file'] = f

def generate_notebook_html(item, lang):
    file_name = item['vn_file'] if lang == 'VN' else item['en_file']
    # Fallback to the other language if the requested one doesn't exist
    if not file_name:
        file_name = item['en_file'] if lang == 'VN' else item['vn_file']
        
    if not file_name:
        return "" # Should not happen
        
    # Format the display title (e.g. 1. The Machine Learning Landscape)
    display_title = f"{item['order']}. {item['display_name']}"
        
    html = f"""    <li style="margin-bottom: 15px; padding: 15px; background: white; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
      <strong style="font-size:16px;">Thực hành: {display_title}</strong><br>
      <div style="margin-top: 10px; display: flex; gap: 10px; flex-wrap: wrap;">
        <a href="{GITHUB_BASE_URL}{file_name}" target="_blank" style="background: #fbbc04; color: #fff; padding: 6px 14px; border-radius: 6px; text-decoration: none; font-size: 14px; font-weight: 600; box-shadow: 0 2px 4px rgba(251,188,4,0.3);">🔥 Mở trên Google Colab</a>
        <a href="{LOCAL_BASE_URL}{file_name}" download style="background: #1a73e8; color: white; padding: 6px 14px; border-radius: 6px; text-decoration: none; font-size: 14px; font-weight: 600; box-shadow: 0 2px 4px rgba(26,115,232,0.3);">💾 Tải file .ipynb về máy</a>
      </div>
    </li>"""
    return html

def update_docs():
    for chap_num, items_dict in chapters.items():
        doc_filename = f"chuong_{chap_num}.md"
        doc_path = os.path.join(DOCS_DIR, doc_filename)
        
        if not os.path.exists(doc_path):
            print(f"Warning: {doc_path} not found.")
            continue
            
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Find the "#### ** 💻 Thực hành **" section
        start_idx = content.find("#### ** 💻 Thực hành **")
        if start_idx == -1:
            print(f"Warning: Practice section not found in {doc_filename}")
            continue
            
        end_idx = content.find("<!-- tabs:end -->", start_idx)
        if end_idx == -1:
            print(f"Warning: tabs:end not found in {doc_filename}")
            continue
            
        # Sort items by order, then by name
        sorted_items = sorted(items_dict.values(), key=lambda x: (x['order'], x['name']))
        
        vn_list_html = "\n".join([generate_notebook_html(item, 'VN') for item in sorted_items])
        en_list_html = "\n".join([generate_notebook_html(item, 'EN') for item in sorted_items])
        
        new_practice_html = f"""#### ** 💻 Thực hành **

<div class="practice-container" style="background: #f8faff; border: 1px solid #cce0ff; border-radius: 8px; padding: 20px; margin-top: 15px;">
  <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
    <h3 style="margin-top:0; color: #1a73e8; display:flex; align-items:center; gap:8px; margin-bottom:0;">🚀 Bài tập Thực hành Jupyter Notebook</h3>
    <div class="lang-toggle" style="display:flex; gap:8px;">
      <button id="btn-vn" onclick="togglePracticeLang('VN')" style="background: #fbbc04; color: #fff; border:none; padding:6px 12px; border-radius:20px; cursor:pointer; font-weight:bold; transition:all 0.2s;">🇻🇳 VN</button>
      <button id="btn-en" onclick="togglePracticeLang('EN')" style="background: #f1f3f4; color: #5f6368; border:none; padding:6px 12px; border-radius:20px; cursor:pointer; font-weight:bold; opacity: 0.4; transition:all 0.2s;">🇬🇧 EN</button>
    </div>
  </div>
  <p style="margin-top: 10px;">Dưới đây là các sổ tay (notebook) chứa mã nguồn Python thực hành cho chương này. Bạn có thể mở trực tiếp trên Google Colab để chạy thử nghiệm, hoặc tải file về máy.</p>
  
  <ul id="notebook-list-VN" style="list-style-type: none; padding-left: 0; display: block;">
{vn_list_html}
  </ul>
  
  <ul id="notebook-list-EN" style="list-style-type: none; padding-left: 0; display: none;">
{en_list_html}
  </ul>

  <div style="margin-top: 20px; border-top: 1px dashed #cce0ff; padding-top: 15px;">
    <strong>Hoặc truy cập toàn bộ kho tài liệu:</strong> <a href="https://drive.google.com/drive/folders/1nRV7W748VkSldg-BaKdcejBV-sBP47_M?usp=sharing" target="_blank" style="color: #1a73e8; font-weight: bold;">Thư mục Google Drive Thực hành</a>
  </div>
</div>

"""
        
        new_content = content[:start_idx] + new_practice_html + content[end_idx:]
        
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"Updated {doc_filename}")

if __name__ == "__main__":
    parse_notebooks()
    update_docs()
    print("Done!")
