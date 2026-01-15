"""
Prompt templates for research agents
Separated into system prompts (role definition) and task prompts (dynamic content)
"""

# ============================================================================
# System Prompts (Agent Role Definitions)
# ============================================================================

PROPOSER_SYSTEM_PROMPT = """You are a creative research proposer. Your role is to generate innovative research ideas or experimental methods based on the given context. Focus on novelty, feasibility, and scientific rigor. Propose specific, actionable ideas that can advance the field."""

PLANNER_SYSTEM_PROMPT = """You are an experimental method planner. Your role is to design coherent, executable experimental workflows based on a given research idea and task description. Focus on ordering of steps, dependencies, roles (RA vs Engineer), and resource planning, ensuring the plan can be realistically implemented."""

CRITIC_SYSTEM_PROMPT = """You are a rigorous research critic. Your role is to identify weaknesses, potential issues, and areas for improvement in research proposals. Provide constructive criticism with specific suggestions. Be thorough but fair, focusing on helping improve the quality of research."""

SEARCHER_SYSTEM_PROMPT = """You are a literature search specialist. Your role is to generate effective search keywords and queries to find relevant academic papers and resources. Consider multiple search strategies and venues."""

SUMMARIZER_SYSTEM_PROMPT = """You are a literature summarization expert. Your role is to extract key information from academic papers and synthesize comprehensive reviews. Focus on themes, methodologies, gaps, and relevant findings."""

ENGINEER_SYSTEM_PROMPT = """You are a software engineer specialized in scientific computing. Your role is to write clean, efficient, and well-documented code for experiments. Follow best practices and ensure reproducibility."""

RESEARCH_ASSISTANT_SYSTEM_PROMPT = """You are a research assistant. Your role is to handle non-coding experimental tasks, data collection, and analysis coordination. Support the research process with careful attention to detail."""

ANALYST_SYSTEM_PROMPT = """You are a data analyst. Your role is to interpret experimental results, identify patterns, and draw meaningful conclusions from data. Be objective and data-driven in your analysis."""

WRITER_SYSTEM_PROMPT = """You are an academic writer. Your role is to compose clear, well-structured research papers following academic standards and conventions. Ensure logical flow and proper citation."""

REVIEWER_SYSTEM_PROMPT = """You are an academic reviewer following ICML standards. Your role is to provide thorough, constructive reviews with clear assessments of strengths and weaknesses. Be fair and help improve the quality of research."""

FORMATTER_SYSTEM_PROMPT = """You are a research formatter and evaluator. Your role is to review debate history, evaluate all proposed ideas or methods, score them based on quality criteria, and format the output as structured content. Be objective and thorough in your evaluation."""

# ============================================================================
# Task Prompts (Dynamic Content Templates)
# ============================================================================

TASK_CLARIFICATION_PROMPT = """Analyze the following research input and determine if it needs clarification:

Input: {input_text}

If the input is clear and contains sufficient information (research question, objectives, constraints), respond with:
CLEAR: [reformatted task description]

If clarification is needed, respond with:
UNCLEAR: [list of specific questions to ask the user]

Focus on:
- Research objectives and scope
- Available resources (data, compute, time)
- Expected outcomes
- Constraints or requirements
"""


LITERATURE_SEARCH_PROMPT = """Generate effective search keywords and queries for the following research task:

Task: {task}

Provide:
1. 3-5 primary search keywords
2. 2-3 search queries for academic databases
3. Suggested venues (conferences, journals)

Format as JSON:
{{
  "keywords": ["keyword1", "keyword2", ...],
  "queries": ["query1", "query2", ...],
  "venues": ["venue1", "venue2", ...]
}}
"""

LITERATURE_SUMMARY_PROMPT = """Synthesize the following literature into a comprehensive review:

Papers:
{papers}

Provide:
1. Key themes and trends
2. Main methodologies used
3. Research gaps identified
4. Relevant findings for the task: {task}

Write in academic style, 300-500 words.
"""

IDEA_PROPOSAL_PROMPT = """Based on the following context, propose an innovative research idea:

Task: {task}
Literature Review: {literature}

Your proposal should include:
1. Core research question
2. Novelty and significance
3. Proposed approach (high-level)
4. Expected contributions

Be specific, feasible, and scientifically rigorous.
"""

IDEA_CRITICISM_PROMPT = """Evaluate the following research idea:

Idea: {idea}

Provide constructive criticism on:
1. Novelty: Is it truly novel or incremental?
2. Feasibility: Can it be realistically executed?
3. Significance: Will it make meaningful contributions?
4. Clarity: Is the idea well-defined?

For each weakness, suggest specific improvements.
"""

METHOD_PROPOSAL_PROMPT = """Design a detailed experimental method for the following research idea:

Idea: {idea}
Task: {task}

Your method should include:
1. Experimental setup and design
2. Step-by-step procedure
3. Task assignments for RA and Engineer
4. Required resources (data, compute, tools)
5. Evaluation metrics

Be specific and actionable.
"""

METHOD_CRITICISM_PROMPT = """Evaluate the following experimental method:

Method: {method}

Assess:
1. Completeness: Are all steps clearly defined?
2. Feasibility: Can it be executed with available resources?
3. Validity: Will it properly test the hypothesis?
4. Efficiency: Are there unnecessary steps?

Suggest specific improvements.
"""

RESULT_ANALYSIS_PROMPT = """Analyze the following experimental results:

Method: {method}
Results: {results}

Provide:
1. Summary of key findings
2. Statistical significance (if applicable)
3. Comparison with expectations
4. Limitations and caveats
5. Implications for the research question

Be objective and data-driven.
"""

PAPER_WRITING_PROMPT = """Write a research paper based on the following:

Task: {task}
Literature: {literature}
Idea: {idea}
Method: {method}
Results: {results}

Structure:
1. Abstract (150-200 words)
2. Introduction (motivation, research question, contributions)
3. Related Work (based on literature review)
4. Methodology (detailed method description)
5. Experiments and Results
6. Discussion
7. Conclusion

Follow academic writing conventions. Use LaTeX format.
"""

REVIEW_PROMPT = """Review the following research paper:

Paper: {paper}

Provide a structured review with:

1. Summary (2-3 sentences)
2. Strengths (3-5 points)
3. Weaknesses (3-5 points)
4. Questions for authors (2-3 questions)
5. Overall Score (1-10, where 10 is best)
6. Confidence (1-5, where 5 is expert)
7. Recommendation (Strong Accept / Accept / Borderline / Reject / Strong Reject)

Be thorough, constructive, and fair.
"""

IDEA_FORMATTER_PROMPT = """Review the following debate history and format the output:

Debate History:
{debate_history}

Task:
1. Extract all proposed ideas from the debate
2. Evaluate each idea based on:
   - Novelty and originality
   - Feasibility and practicality
   - Scientific rigor
   - Potential impact
3. Assign a quality score (0.0-1.0) to each idea
4. Rank ideas by score (best first)

Output format (JSON):
{{
  "ideas": [
    {{
      "content": "detailed idea description",
      "score": 0.95,
      "round": 3,
      "strengths": ["strength1", "strength2"],
      "weaknesses": ["weakness1", "weakness2"]
    }},
    ...
  ]
}}
"""

METHOD_FORMATTER_PROMPT = """Review the following debate history and format the output:

Debate History:
{debate_history}

Task:
1. Extract the final experimental method from the debate
2. Structure it with:
   - Overview
   - Detailed steps
   - Task assignments (RA and Engineer)
   - Required resources
3. Include criticisms and how they were addressed

Output format (JSON):
{{
  "overview": "method overview",
  "steps": ["step1", "step2", ...],
  "assignments": [
    {{"role": "Engineer", "tasks": ["task1", "task2"]}},
    {{"role": "RA", "tasks": ["task1", "task2"]}}
  ],
  "resources": {{"data": "...", "compute": "..."}},
  "debate_rounds": 3
}}
"""
