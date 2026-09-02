import os

def main():
    path = 'index.html'
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    css = '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">'
    js = '<!-- Docsify KaTeX -->\n  <script src="https://cdn.jsdelivr.net/npm/docsify-katex@1/dist/docsify-katex.js"></script>'

    # Inject CSS before </head>
    if css not in text:
        text = text.replace('</head>', f'  {css}\n</head>')

    # Inject JS before <!-- Search Plugin -->
    if 'docsify-katex.js' not in text:
        text = text.replace('<!-- Search Plugin -->', f'{js}\n\n  <!-- Search Plugin -->')

    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)

    print("Success")

if __name__ == "__main__":
    main()
