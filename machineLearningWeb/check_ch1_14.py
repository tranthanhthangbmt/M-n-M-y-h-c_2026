import os, json, re
filepath = r'd:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Máy học_2026\machineLearningWeb\quizzes\Chapter01\index.html'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()
match = re.search(r'const questions = (\[.*?\]);\n\n    let currentQ = 0;', content, flags=re.DOTALL)
questions = json.loads(match.group(1))
q = questions[13] # Index 13
correct_id = q.get('correctAnswer')
options = q.get('options', [])
correct_opt = next((opt for opt in options if opt['id'] == correct_id), None)
print('Length:', len(correct_opt['text']))
print('Text:', correct_opt['text'])
