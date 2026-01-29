import re
from pathlib import Path

# -------------------------
# 基础 LaTeX -> Markdown 转换
# -------------------------
def latex_to_md(text: str) -> str:
    rules = [
        (r'\\section\{(.+?)\}', r'## \1\n'),
        (r'\\subsection\{(.+?)\}', r'### \1\n'),
        (r'\\subsubsection\{(.+?)\}', r'#### \1\n'),
        (r'\\textbf\{(.+?)\}', r'**\1**'),
        (r'\\emph\{(.+?)\}', r'*\1*'),
        (r'\\cite\{(.+?)\}', r'[@\1]'),
        (r'\\ref\{(.+?)\}', r'(ref: \1)'),
        (r'\\item', r'-'),
    ]

    for pattern, repl in rules:
        text = re.sub(pattern, repl, text, flags=re.DOTALL)

    # 去掉常见环境标记
    text = re.sub(r'\\begin\{.*?\}', '', text)
    text = re.sub(r'\\end\{.*?\}', '', text)

    return text.strip()


# -------------------------
# 解析主 LaTeX 文件
# -------------------------
def parse_main_tex(tex_path: Path) -> str:
    content = tex_path.read_text(encoding="utf-8")

    md_parts = []

    # Title
    title_match = re.search(r'\\icmltitle\{(.+?)\}', content)
    if title_match:
        md_parts.append(f"# {title_match.group(1)}\n")

    # Abstract (JSON 形式)
    abs_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', content, re.S)
    if abs_match:
        abstract = abs_match.group(1)
        abstract = re.sub(r'[\{\}]', '', abstract)
        md_parts.append("## Abstract\n")
        md_parts.append(abstract.strip() + "\n")

    # Keywords
    kw_match = re.search(r'\\icmlkeywords\{(.+?)\}', content)
    if kw_match:
        md_parts.append("**Keywords:** " + kw_match.group(1) + "\n")

    # Sections via \input
    inputs = re.findall(r'\\input\{(.+?)\}', content)
    for section in inputs:
        sec_path = tex_path.parent / section
        if sec_path.exists():
            sec_text = sec_path.read_text(encoding="utf-8")
            md_parts.append(latex_to_md(sec_text) + "\n")
        else:
            md_parts.append(f"> ⚠ Missing file: {section}\n")

    # References 占位
    md_parts.append("## References\n")
    md_parts.append("> Bibliography converted separately (BibTeX → CSL recommended)\n")

    return "\n".join(md_parts)


# -------------------------
# 主入口
# -------------------------
if __name__ == "__main__":
    main_tex = Path("paper/main.tex")
    output_md = Path("paper/output.md")

    markdown = parse_main_tex(main_tex)
    output_md.write_text(markdown, encoding="utf-8")

    print(f"Markdown generated at: {output_md}")
