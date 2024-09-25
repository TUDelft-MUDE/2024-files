import markdown

with open('crossword.md', 'r') as f:
    text = f.read()
    html = markdown.markdown(text,extensions=['markdown.extensions.tables','sane_lists'])

with open('auxiliary_files/crossword.html', 'w') as f:
    f.write(html)