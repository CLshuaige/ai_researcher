"""LaTeX template presets for different journal formats"""
from typing import Callable
from pathlib import Path
from enum import Enum


class Journal(str, Enum):
    """Supported journal formats"""
    ICML2026 = "ICML2026"
    """ICML 2026 - International Conference on Machine Learning"""


class LatexPreset:
    """LaTeX preset configuration for a journal"""
    def __init__(
        self,
        article: str,
        layout: str = "",
        title: str = r"\title",
        author: Callable[[str], str] = lambda x: f"\\author{{{x}}}",
        affiliation: Callable[[str], str] = lambda x: f"\\affiliation{{{x}}}",
        abstract: Callable[[str], str] = lambda x: f"\\begin{{abstract}}\n{x}\n\\end{{abstract}}",
        keywords: Callable[[str], str] = lambda x: "",
        bibliographystyle: str = r"\bibliographystyle{plain}",
        usepackage: str = "",
        files: list[str] = None,
    ):
        self.article = article
        self.layout = layout
        self.title = title
        self.author = author
        self.affiliation = affiliation
        self.abstract = abstract
        self.keywords = keywords
        self.bibliographystyle = bibliographystyle
        self.usepackage = usepackage
        self.files = files or []


# ICML2026 preset
icml2026_preset = LatexPreset(
    article="article",
    title=r"\twocolumn[\n\icmltitle",
    author=lambda x: f"\\begin{{icmlauthorlist}}\n\\icmlauthor{{{x}}}{{aff}}\n\\end{{icmlauthorlist}}",
    affiliation=lambda x: f"\\icmlaffiliation{{aff}}{{{x}}}\n",
    abstract=lambda x: f"]\n\\printAffiliationsAndNotice{{}}\n\\begin{{abstract}}\n{x}\n\\end{{abstract}}\n",
    keywords=lambda x: f"\\icmlkeywords{{{x}}}" if x else "",
    bibliographystyle=r"\bibliographystyle{icml2026}",
    usepackage=r"\usepackage[accepted]{icml2026}",
    files=['icml2026.sty', 'icml2026.bst', 'fancyhdr.sty', 'algorithm.sty', 'algorithmic.sty'],
)


journal_presets = {
    Journal.ICML2026: icml2026_preset,
}


def get_preset(journal: Journal) -> LatexPreset:
    """Get LaTeX preset for a journal"""
    return journal_presets.get(journal, icml2026_preset)
