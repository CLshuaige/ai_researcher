import os
import os.path as osp
import shutil
import subprocess
import sys
from typing import Tuple

import re
import tempfile
import time
import traceback
import unicodedata
from typing import Optional, Tuple, List, Dict
from pathlib import Path


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

def compile_latex(work_dir: Path, tex_content: str, tex_path: Path, sample_name: str) -> Tuple[bool, str]:
    """
    编译LaTeX文件

    Returns:
        (是否成功, PDF路径或错误信息)
    """

    # 写入文件
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_content)

    print(f"  已写入: {tex_path}")

    # 检查是否需要运行bibtex（检测是否有cite命令或bibliography）
    has_citations = "\\cite{" in tex_content or "\\bibliography{" in tex_content

    # 编译命令序列
    commands = [
        ["pdflatex", "-interaction=nonstopmode", f"{sample_name}.tex"],
    ]

    # 只有存在引用时才运行bibtex
    if has_citations:
        commands.append(["bibtex", sample_name])

    # 后续pdflatex运行
    commands.extend([
        ["pdflatex", "-interaction=nonstopmode", f"{sample_name}.tex"],
        ["pdflatex", "-interaction=nonstopmode", f"{sample_name}.tex"],
    ])

    pdf_path = osp.join(".", f"{sample_name}.pdf")

    for i, cmd in enumerate(commands):
        try:
            print(f"  执行: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                print(f"    警告: 命令返回非零状态 {result.returncode}")
                # pdflatex的错误通常可以继续
                if "pdflatex" in cmd[0]:
                    print(f"    继续编译...")

        except subprocess.TimeoutExpired:
            return False, "编译超时"
        except FileNotFoundError as e:
            return False, f"命令未找到: {cmd[0]} - {e}"
        except Exception as e:
            return False, f"编译异常: {str(e)}"

    # 移动PDF
    generated_pdf = osp.join(work_dir, f"{sample_name}.pdf")
    if osp.exists(generated_pdf):
        shutil.copy(generated_pdf, pdf_path)
        return True, pdf_path
    else:
        return False, "PDF文件未生成"


# ========== 测试代码 ==========
if __name__ == "__main__":
    import tempfile
    import shutil

    # 路径设置
    template_path = "./researcher/latex/literature/template.tex"

    # 创建临时工作目录
    work_dir = Path(tempfile.mkdtemp(prefix="latex_test_"))
    output_dir = Path(tempfile.mkdtemp(prefix="latex_output_"))

    print(f"工作目录: {work_dir}")
    print(f"输出目录: {output_dir}")

    # 读取模板文件
    with open(template_path, "r", encoding="utf-8") as f:
        tex_content = f.read()

    print(f"已读取模板: {template_path} ({len(tex_content)} 字符)")

    # 设置 tex 文件路径
    tex_path = work_dir / "test_literature.tex"

    # 调用 compile_latex
    success, result = compile_latex(work_dir, tex_content, tex_path, "test_literature")

    if success:
        print(f"\n✅ 编译成功!")
        print(f"PDF 路径: {result}")
    else:
        print(f"\n❌ 编译失败!")
        print(f"错误信息: {result}")

    # 清理临时目录（可选，注释掉以保留文件检查）
    # shutil.rmtree(work_dir)
    # shutil.rmtree(output_dir)
    # print(f"\n已清理临时目录")
