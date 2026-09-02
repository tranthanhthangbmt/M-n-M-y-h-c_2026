import re
text = r'\\[text\\]'
res = re.sub(r'\\\\\[(.*?)\\\\\]', r'$$\1$$', text, flags=re.DOTALL)
print('Result:', res)
