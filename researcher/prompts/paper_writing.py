# ============================================================================
# 1. Outline Generation
# ============================================================================

def outline_prompt(task: str, idea: str, method_summary: str, results_summary: str) -> str:
    """Generate detailed outline for research paper"""
    return f"""Given the research context below, generate a comprehensive paper outline in JSON format.

Research Task:
{task}

Research Idea:
{idea}

Method Summary:
{method_summary}

Results Summary:
{results_summary}

Generate a structured outline with the following requirements:

**Sections**: List of sections with:
   - **id**: Sequential number (1, 2, 3, ...)
   - **title**: Section title (e.g., "Introduction", "Related Work", "Methodology", "Experiments", "Results", "Discussion", "Conclusion")
   - **description**: Detailed description of what should be covered in this section

Follow ICML2026 paper structure conventions:
- Introduction should establish context and motivation
- Related Work (if included) should position this work
- Methodology should describe the approach in detail
- Experiments should describe setup and evaluation
- Results should present findings with analysis
- Discussion/Conclusion should summarize contributions and limitations

**Output Format (JSON only)**:
{{
  "sections": [
    {{
      "id": 1,
      "title": "Introduction",
      "description": "Detailed description of what to cover",
      "guidelines": "Advanced guidelines for the section"
    }},
    {{
      "id": 2,
      "title": "Related Work",
      "description": "Detailed description of what to cover",
      "guidelines": "Advanced guidelines for the section"
    }}
  ]
}}

Respond with valid JSON only, no additional text."""


# ============================================================================
# 2. Abstract Generation
# ============================================================================

def abstract_initial_prompt(idea: str, method_summary: str, results_summary: str, 
                            outline_sections: str) -> str:
    """Generate abstract and title for research paper (initial generation)"""
    return f"""Given the research context below, generate a title and abstract for a research paper following ICML2026 format.

Research Idea:
{idea}

Method Summary:
{method_summary}

Results Summary:
{results_summary}

Paper Outline:
{outline_sections}

Guidelines:
- Generate a clear, concise title that reflects the main contribution
- Abstract should be a single paragraph, no line breaks or sections
- 150-250 words
- Cover: problem statement, approach, key results, contributions
- No equations or citations in abstract

**Output Format (JSON only)**:
{{
  "title": "Paper Title",
  "abstract": "Abstract text in LaTeX format (without \\begin{{abstract}} wrapper)"
}}

Respond with valid JSON only, no additional text."""


# ============================================================================
# 3. Abstract Reflection/Improvement
# ============================================================================

def abstract_prompt(title: str, abstract: str, idea: str, 
                   method_summary: str, results_summary: str, output_type: str = "latex") -> str:
    """Generate prompt for improving abstract through self-reflection"""
    if output_type == "latex":
        #format_desc = "LaTeX format following ICML2026 standards"
        output_block = (
            "\\begin{Abstract}\n"
            "<ABSTRACT>\n"
            "\\end{Abstract}"
        )
        content_instruction = (
            "In <ABSTRACT>, place the improved abstract in valid LaTeX format. "
        )

    elif output_type == "markdown":  # markdown
        #format_desc = "Markdown format suitable for academic papers"
        output_block = (
            "## Abstract\n\n"
            "<ABSTRACT>"
        )
        content_instruction = (
            "In <ABSTRACT>, place the abstract in Markdown format. "
        )
    return f"""Rewrite the abstract below to make it more clear and well-motivated. You are given the idea, methods, and results of the paper together with the previously written abstract.

Paper Title:
{title}

Previous Abstract:
{abstract}

Research Idea:
{idea}

Method Summary:
{method_summary}

Results Summary:
{results_summary}

Guidelines:
- Abstract should be a single paragraph, no sections, subsections, or breaks between lines
- Briefly describe the problem
- Briefly describe how we try to solve it
- Mention the dataset and methods used
- Briefly describe the results
- Please make sure the abstract reads smoothly and is well-motivated
- Do not add \\begin{{abstract}} or \\end{{abstract}} wrappers

**Output Format**:
{output_block}

{content_instruction}"""


# ============================================================================
# 4. Keywords Extraction
# ============================================================================

def keywords_prompt(abstract: str, title: str) -> str:
    """Generate prompt for extracting keywords from abstract and title"""
    return f"""Given the paper title and abstract below, extract 3-8 relevant keywords for a research paper.

Paper Title:
{title}

Paper Abstract:
{abstract}

Guidelines:
- Extract 3-8 keywords that best represent the core concepts, techniques, and contributions mentioned in the abstract
- Keywords should be specific and relevant to the research
- Use comma-separated format
- Each keyword should be a single word or short phrase (2-3 words max)
- Keywords should reflect the main topics, methods, and contributions described in the abstract

**Output Format (JSON only)**:
{{
  "keywords": "keyword1, keyword2, keyword3"
}}

Respond with valid JSON only, no additional text."""


# ============================================================================
# 5. Section Writing
# ============================================================================

def introduction_prompt(title: str, abstract: str, idea: str, method_summary: str, output_type: str = "latex") -> str:
    """Generate introduction section"""
    if output_type == "latex":
        format_desc = "LaTeX format following ICML2026 standards"
        output_block = (
            "\\begin{Introduction}\n"
            "<INTRODUCTION_TEXT>\n"
            "\\end{Introduction}"
        )
        content_instruction = (
            "In <INTRODUCTION_TEXT>, place the introduction content in valid LaTeX format. "
        )

    elif output_type == "markdown":  # markdown
        format_desc = "Markdown format suitable for academic papers"
        output_block = (
            "## Introduction\n\n"
            "<INTRODUCTION_TEXT>"
        )
        content_instruction = (
            "In <INTRODUCTION_TEXT>, place the introduction content in Markdown format. "
        )
    return f"""Write an introduction section for a research paper in {format_desc}.

Paper Title:
{title}

Paper Abstract:
{abstract}

Research Idea:
{idea}

Method Summary:
{method_summary}

Guidelines:
- Expand on key points from abstract with more background and context
- Describe the problem and why it is difficult/important
- Explain how this paper attempts to solve it
- Describe how results verify the solution
- Do NOT create subsections (this is a section, not a document)
- Do NOT add citations (citations will be added later)

**Output Format**:
{output_block}

{content_instruction}"""


def methods_prompt(title: str, abstract: str, introduction: str, method_summary: str, output_type: str = "latex") -> str:
    """Generate methods section"""
    if output_type == "latex":
        format_desc = "LaTeX format following ICML2026 standards"
        output_block = (
            "\\begin{Methods}\n"
            "<METHODS_TEXT>\n"
            "\\end{Methods}"
        )
        content_instruction = (
            "In <METHODS_TEXT>, place the methods content in valid LaTeX format. "
        )

    elif output_type == "markdown":  # markdown
        format_desc = "Markdown format suitable for academic papers"
        output_block = (
            "## Methods\n\n"
            "<METHODS_TEXT>"
        )
        content_instruction = (
            "In <METHODS_TEXT>, place the methods content in Markdown. "
        )
    return f"""Write the methods section for a research paper in {format_desc}.

Paper Title:
{title}

Paper Abstract:
{abstract}

Paper Introduction:
{introduction}

Method Summary:
{method_summary}

Guidelines:
- Describe in detail the different methods, datasets, evaluation metrics, and relevant elements
- Do NOT write citations (references will be added later)
- Connect the text with the introduction section
- You can create subsections and subsubsections, but NOT sections (this is already a section)

**Output Format**:
{output_block}

{content_instruction}"""


def results_prompt(title: str, abstract: str, introduction: str, methods: str, 
                  results_summary: str, output_type: str = "latex") -> str:
    """Generate results section"""
    if output_type == "latex":
        format_desc = "LaTeX format following ICML2026 standards"
        output_block = (
            "\\begin{Results}\n"
            "<RESULTS_TEXT>\n"
            "\\end{Results}"
        )
        content_instruction = (
            "In <RESULTS_TEXT>, place the results content in valid LaTeX format. "
        )

    elif output_type == "markdown":  # markdown
        format_desc = "Markdown format suitable for academic papers"
        output_block = (
            "## Results\n\n"
            "<RESULTS_TEXT>"
        )
        content_instruction = (
            "In <RESULTS_TEXT>, place the results content in Markdown. "
        )
    return f"""Write the results section for a research paper in {format_desc}.

Paper Title:
{title}

Paper Abstract:
{abstract}

Paper Introduction:
{introduction}

Paper Methods:
{methods}

Results Summary:
{results_summary}

Guidelines:
- Explain carefully the results obtained
- Do NOT add plots or placeholders for plots (figures will be added later)
- Describe what we learned from the results
- Do NOT write bibliography
- You can create subsections and subsubsections, but NOT sections
- The paper uses two-column format, so for long equations and wide tables, use full page width or single column
- Connect the text with introduction and methods sections
- You can summarize results at the end, but do NOT write a conclusions subsection

**Output Format**:
{output_block}

{content_instruction}"""


def conclusions_prompt(title: str, abstract: str, introduction: str, methods: str, 
                      results: str, output_type: str = "latex") -> str:
    """Generate conclusions section"""
    if output_type == "latex":
        format_desc = "LaTeX format following ICML2026 standards"
        output_block = (
            "\\begin{Conclutions}\n"
            "<CONCLUTIONS_TEXT>\n"
            "\\end{Conclutions}"
        )
        content_instruction = (
            "In <CONCLUTIONS_TEXT>, place the conclutions content in valid LaTeX format. "
        )

    elif output_type == "markdown":  # markdown
        format_desc = "Markdown format suitable for academic papers"
        output_block = (
            "## Conclutions\n\n"
            "<CONCLUTIONS_TEXT>"
        )
        content_instruction = (
            "In <CONCLUTIONS_TEXT>, place the conclutions content in Markdown. "
        )
    return f"""Write the conclusions section for a research paper in {format_desc}.

Paper Title:
{title}

Paper Abstract:
{abstract}

Paper Introduction:
{introduction}

Paper Methods:
{methods}

Paper Results:
{results}

Guidelines:
- Briefly describe the problem and how this paper tries to solve it
- Describe the datasets and methods used
- Describe the results obtained
- Describe what we learned from the results and this paper
- Do NOT add citations (citations will be added later)
- Do NOT write words or sentences between asterisks (*)

**Output Format**:
{output_block}

{content_instruction}"""


def section_prompt(section_id: int, section_title: str, section_description: str, section_guidelines: str,
                  title: str, abstract: str, completed_sections: str,
                  task: str, idea: str, method_summary: str, results_summary: str, output_type: str = "latex") -> str:
    """Generate prompt for writing a specific section (generic fallback)"""
    if output_type == "latex":
        format_desc = "LaTeX format following ICML2026 standards"
        output_block = (
            f"\\begin{{{section_title}}}\n"
            "<SECTION_TEXT>\n"
            f"\\end{{{section_title}}}"
        )
        content_instruction = (
            f"In <SECTION_TEXT>, place the section content in valid LaTeX format. "
        )

    elif output_type == "markdown":  # markdown
        format_desc = "Markdown format suitable for academic papers"
        output_block = (
            f"## {{{section_title}}}\n\n"
            "<SECTION_TEXT>"
        )
        content_instruction = (
            "In <SECTION_TEXT>, place the section content in Markdown. "
        )
    return f"""Write the "{section_title}" section for a research paper in {format_desc}.
You are provided with these information and previous sections:

Section ID: {section_id}
Section Title: {section_title}
Section Description:
{section_description}

Paper Title:
{title}

Paper Abstract:
{abstract}

Research Task:
{task}

Research Idea:
{idea}

Method Summary:
{method_summary}

Results Summary:
{results_summary}

Previous Sections (for context):
{completed_sections}

Guidelines:
{section_guidelines}

**Output Format**:
{output_block}

{content_instruction}"""


# ============================================================================
# 6. Citation Addition
# ============================================================================

def citation_addition_prompt(paragraph: str) -> str:
    """Generate prompt for adding citations to a paragraph using Perplexity API"""
    return f"""Search for relevant academic papers on arXiv that support the claims in the following paragraph. Add citations in numerical format [1], [2], [3], etc. where appropriate.

Paragraph:
{paragraph}

Guidelines:
- Only search on arXiv (arxiv.org)
- Add citations only where they are clearly relevant and necessary to support claims
- Use numerical format: [1], [2], [3] for single citations, [1][2][3] for multiple citations
- Do not add citations to very short paragraphs or unclear statements
- Do not modify the original text, only add citation markers
- Do not add citations inside tables, figures, or equations
- Return the paragraph with citations added, without any additional text or explanations

Return only the paragraph with citations added, no other text."""


# ============================================================================
# 7. LaTeX Error Fixing
# ============================================================================

def fix_latex_prompt(section_text: str, error_message: str, section_name: str) -> str:
    """Generate prompt for fixing LaTeX compilation errors"""
    return f"""The LaTeX text below has compilation errors. Your task is to fix the text so that it compiles properly in LaTeX.

Section Name: {section_name}

LaTeX Compilation Error:
{error_message}

Original Text:
{section_text}

Guidelines:
- The text you are given is just a section of a LaTeX paper, not a complete document
- Fix **all LaTeX errors** found in the compilation error
- Pay special attention to underscores: if an underscore _ is outside math mode, it may need to be \\_ to compile properly
- Pay special attention to % symbols: they must be escaped as \\% outside of comments
- Return the original text but with the errors fixed
- Keep the text intact. Only fix the errors without changing anything else
- Do not add document structure (\\begin{{document}}, etc.)

**Output Format**:
\\begin{{Text}}
<TEXT>
\\end{{Text}}

In <TEXT>, put the fixed version of the text with all LaTeX errors corrected."""
