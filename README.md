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

```bash
#LaTex
conda install anaconda::poppler      # pdftotext/pdfinfo                                                                                                              
conda install conda-forge::chktex    # LaTeX语法检查                                                                                                                  
sudo apt install texlive-latex-extra       # LaTeX编译环境
```

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Edit `.env` with your API keys and preferences.

## Usage

### CLI

```bash
python -m researcher.main --input "Your research task" --project-name "my_project"
```

Or with input file:

```bash
python -m researcher.main --input path/to/input.md --project-name "my_project"
```

### FastAPI

Start API server:

```bash
python api_main.py
```

Default address: `http://127.0.0.1:8001`

OpenAPI docs:

- `http://127.0.0.1:8001/docs`
- `http://127.0.0.1:8001/redoc`

API endpoints:

- `GET /health`: health check.
- `POST /api/v1/projects`: create project.
- `GET /api/v1/projects`: list all projects.
- `GET /api/v1/projects/latest`: get latest updated project.
- `POST /api/v1/projects/{project_id}/config`: patch project config.
- `POST /api/v1/projects/{project_id}/runs`: run workflow.
- `GET /api/v1/projects/{project_id}`: project status.
- `GET /api/v1/projects/{project_id}/nodes/{node_name}/latest`: latest node result.
- `GET /api/v1/projects/{project_id}/files`: list all files in workspace; use `?download=true` to download whole project zip.
- `GET /api/v1/projects/{project_id}/files/{file_path}`: read file; use `?download=true` to download single file.
- `PUT /api/v1/projects/{project_id}/files/{file_path}`: upload/create/update file (`utf-8` or `base64`).
- `GET /api/v1/projects/{project_id}/history/{node_name}`: node history files.
- `GET /api/v1/projects/{project_id}/logs?tail_lines=200`: logs.
- `WS /api/v1/projects/{project_id}/events`: realtime events.
- `POST /api/v1/projects/{project_id}/input`: user input request.

### Local Files Upload for Source Ingestion

Upload these files to `input/` via `PUT /api/v1/projects/{project_id}/files/{file_path}`.

1. `input/sources_url.json`: declare URL/git sources.

```json
{
  "sources": [
    "https://example.com/spec.pdf",
    "https://github.com/org/repo.git",
    "input/local_file.md"
  ]
}
```

2. `input/source_annotations.json`: optional corresponding per-source note.

```json
{
  "items": [
    {
      "source": "input/local_file.md",
      "note": "Core project context, keep important constraints."
    },
    {
      "source": "https://example.com/spec.pdf",
      "note": "Focus on requirements and edge cases."
    }
  ]
}
```
