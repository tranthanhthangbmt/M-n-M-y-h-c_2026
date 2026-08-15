import os
import json
import re

QUIZ_DIR = r"d:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Máy học_2026\machineLearningWeb\quizzes"

broken = []

for i in range(1, 20):
    filepath = os.path.join(QUIZ_DIR, f'Chapter{i:02d}', 'index.html')
    if not os.path.exists(filepath): continue
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    match = re.search(r'const questions = (\[.*?\]);\n\n    let currentQ = 0;', content, flags=re.DOTALL)
    if not match: continue
    
    questions = json.loads(match.group(1))
    for q_idx, q in enumerate(questions):
        if q.get('type') == 'multiple_choice':
            correct_id = q.get('correctAnswer')
            options = q.get('options', [])
            correct_opt = next((opt for opt in options if opt['id'] == correct_id), None)
            
            if correct_opt and '...' in correct_opt['text']:
                broken.append({
                    'chap': i,
                    'q_idx': q_idx,
                    'question': q['question'],
                    'old_text': correct_opt['text'],
                    'explanation': q.get('explanation', '')[:100] + '...' # Just for context
                })

with open('broken.json', 'w', encoding='utf-8') as f:
    json.dump(broken, f, ensure_ascii=False, indent=2)

print(f"Extracted {len(broken)} broken questions to broken.json")
