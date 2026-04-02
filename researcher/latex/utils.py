import subprocess
from typing import Tuple, Optional
from pathlib import Path
import re


def extract_latex_code(response: str) -> Optional[str]:
    """
    从LLM响应中提取LaTeX代码

    支持格式:
    ```latex
    ...
    ```
    或
    ```
    ...
    ```
    """
    # 尝试匹配 ```latex 代码块
    pattern = r"```latex\s*(.*?)\s*```"
    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # 尝试匹配普通代码块
    pattern = r"```\s*(.*?)\s*```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        content = match.group(1).strip()
        # 检查是否包含LaTeX特征
        if "\\documentclass" in content or "\\begin{document}" in content:
            return content

    # 如果没有代码块，检查整个文本
    if "\\documentclass" in response or "\\begin{document}" in response:
        # 提取documentclass到end{document}之间的内容
        start = response.find("\\documentclass")
        if start == -1:
            start = response.find("\\begin{document}")
        end = response.rfind("\\end{document}")
        if end != -1:
            end += len("\\end{document}")
            return response[start:end].strip()

    return None

def compile_latex(tex_path: Path, tex_content: str) -> Tuple[bool, str]:
    """
    编译LaTeX文件

    Args:
        tex_path: tex文件路径（包含目录和文件名）
        tex_content: LaTeX内容

    Returns:
        (是否成功, PDF路径或错误信息)
    """
    work_dir = tex_path.parent
    sample_name = tex_path.stem

    # 清理可能冲突的旧辅助文件（filecontents需要生成新的references.bib）
    aux_files = [".aux", ".bbl", ".blg", ".log", ".out", ".toc"]
    for ext in aux_files:
        f = work_dir / f"{sample_name}{ext}"
        if f.exists():
            f.unlink()
    # 删除可能存在的旧references.bib（让filecontents重新生成）
    old_bib = work_dir / "references.bib"
    if old_bib.exists():
        old_bib.unlink()

    # 写入文件
    tex_path.write_text(tex_content, encoding="utf-8")
    print(f"  已写入: {tex_path}")

    # 检查是否需要运行bibtex
    has_citations = "\\cite{" in tex_content or "\\bibliography{" in tex_content

    # 编译命令序列
    commands = [["pdflatex", "-interaction=nonstopmode", f"{sample_name}.tex"]]
    if has_citations:
        commands.append(["bibtex", sample_name])
    commands.extend([
        ["pdflatex", "-interaction=nonstopmode", f"{sample_name}.tex"],
        ["pdflatex", "-interaction=nonstopmode", f"{sample_name}.tex"],
    ])

    for cmd in commands:
        try:
            print(f"  执行: {' '.join(cmd)}")
            result = subprocess.run(
                cmd, cwd=work_dir, capture_output=True, timeout=60
            )
            # 处理可能的编码问题
            stdout = result.stdout.decode('utf-8', errors='replace') if result.stdout else ""
            stderr = result.stderr.decode('utf-8', errors='replace') if result.stderr else ""

            if result.returncode != 0 and "pdflatex" in cmd[0]:
                print(f"    警告: 编译返回非零状态，继续...")
                if stderr:
                    print(f"    stderr: {stderr[:500]}")
        except subprocess.TimeoutExpired:
            return False, "编译超时"
        except FileNotFoundError as e:
            return False, f"命令未找到: {cmd[0]} - {e}"
        except Exception as e:
            return False, f"编译异常: {str(e)}"

    # 检查PDF是否生成
    pdf_path = work_dir / f"{sample_name}.pdf"
    return (True, str(pdf_path)) if pdf_path.exists() else (False, "PDF文件未生成")


# ========== 测试代码 ==========
if __name__ == "__main__":
    import tempfile

    # 路径设置
    template_path = "/home/ai_researcher/projects/ai_researcher/workspace/api_projects/20260401_192858_neural-operator_a7e6c8f0/literature.tex"

    # 创建临时工作目录
    work_dir = Path(tempfile.mkdtemp(prefix="latex_test_"))

    print(f"工作目录: {work_dir}")

    # 读取模板文件
    with open(template_path, "r", encoding="utf-8") as f:
        tex_content = f.read()

    print(f"已读取模板: {template_path} ({len(tex_content)} 字符)")

    # 设置 tex 文件路径
    tex_path = "/home/ai_researcher/projects/ai_researcher/workspace/api_projects/20260401_192858_neural-operator_a7e6c8f0/literature.tex"
    tex_path = Path(tex_path)

    # 调用 compile_latex
    success, result = compile_latex(tex_path, tex_content)

    if success:
        print(f"\n✅ 编译成功!")
        print(f"PDF 路径: {result}")
    else:
        print(f"\n❌ 编译失败!")
        print(f"错误信息: {result}")

    # 清理临时目录（可选，注释掉以保留文件检查）
    # import shutil
    # shutil.rmtree(work_dir)
    # print(f"\n已清理临时目录")
