# AI Researcher

Modular scientific research automation system powered by LangGraph and AG2.

## Architecture

- **LangGraph**: Manages global state flow and macro-level workflow control
- **AG2 (AutoGen)**: Handles multi-agent collaboration and debate within specific nodes
- **Python 3.13**: Core implementation language

## Workflow

1. **Task Parsing**: Clarify research objectives with optional human-in-the-loop
2. **Literature Review**: Search and synthesize relevant literature
3. **Hypothesis Construction**: Generate research ideas through agent debate
4. **Method Design**: Design experimental methods through agent debate
5. **Experiment Execution**: Execute experiments and analyze results
6. **Report Generation**: Generate research paper
7. **Review**: Provide ICML-style peer review

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Edit `.env` with your API keys and preferences.

## Usage

### Command Line

```bash
python -m researcher.main --input "Your research task" --project-name "my_project"
```

Or with input file:

```bash
python -m researcher.main --input path/to/input.md --project-name "my_project"
```

### Web Interface

Launch the Gradio web interface:

```bash
python gradio.py
```

Then open your browser to `http://localhost:7860` to access the interactive UI with:
- Model selection and configuration
- Real-time workflow progress tracking
- Artifact visualization
- Development status overview

## Structure

```
researcher/
├── config.py              # Configuration management
├── state.py               # LangGraph state schema
├── schemas.py             # Pydantic data models
├── utils.py               # Utility functions
├── llm.py                 # LLM client and model presets
├── exceptions.py          # Custom exceptions
├── debate.py              # AG2 debate orchestration
├── researcher.py          # Main researcher class
├── main.py                # Entry point
├── graph/
│   └── researcher_graph.py  # LangGraph workflow definition
├── agents/                # AG2 agent implementations
│   ├── proposer.py
│   ├── critic.py
│   ├── formatter.py
│   ├── searcher.py
│   ├── summarizer.py
│   ├── writer.py
│   └── reviewer.py
├── nodes/                 # Node implementations
│   ├── task_parsing.py
│   ├── literature_review.py
│   ├── hypothesis_construction.py
│   ├── method_design.py
│   ├── experiment_execution.py
│   ├── report_generation.py
│   └── review.py
└── prompts/
    └── templates.py       # Prompt templates

workspace/                 # Generated research artifacts
└── {timestamp}_{project}/
    ├── input.md
    ├── task.md
    ├── literature.md
    ├── literature/
    │   ├── papers.json
    │   └── arxiv_cache/
    │       ├── {timestamp}_{arxiv_id}.pdf
    │       └── {timestamp}_metadata.json
    ├── idea.md
    ├── method.md
    ├── results.md
    ├── paper.pdf
    ├── referee.md
    ├── code/
    ├── data/
    ├── figures/
    ├── tex/
    └── logs/
        ├── debate_hypothesis_{timestamp}.json
        └── debate_method_{timestamp}.json
```

## Development Status

**Preliminary completion:**
- Core framework with LangGraph workflow orchestration
- AG2-based multi-agent debate system (DebateTeam)
- All 7 workflow nodes with structured data models
- Literature review with arXiv integration and PDF caching
- Hypothesis construction through Proposer-Critic debate
- Method design through Proposer-Critic debate
- Report generation with Writer agent
- ICML-style review with Reviewer agent
- Model presets for OpenAI, Anthropic, and local models
- Workspace management with timestamped directories

**TODO:**
- Human-in-the-loop logic for task parsing (Asker + Formatter agents)
- AG2 multi-agent collaboration for experiment execution (RA + Engineer)
- Code execution environment for experiments
- LaTeX compilation for paper generation

## License

MIT
