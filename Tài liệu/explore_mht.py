import email
from email import policy
import os
import sys

mht_file = r"d:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Máy học_2026\Tài liệu\Hands-On Machine Learning with ScikitLearn, Keras, and TensorFlow.mht"
output_dir = r"d:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Máy học_2026\Tài liệu\scratch"

os.makedirs(output_dir, exist_ok=True)

print(f"Reading {mht_file}...")
try:
    with open(mht_file, 'rb') as f:
        msg = email.message_from_binary_file(f, policy=policy.default)

    print("Extracting parts...")
    for part in msg.walk():
        content_type = part.get_content_type()
        print(f"Found part: {content_type}")
        
        if content_type == 'text/html':
            html_content = part.get_payload(decode=True)
            with open(os.path.join(output_dir, 'extracted.html'), 'wb') as html_out:
                html_out.write(html_content)
            print(f"Saved extracted.html, length: {len(html_content)} bytes")
            break # Just extract the HTML for now
except Exception as e:
    print(f"Error: {e}")
