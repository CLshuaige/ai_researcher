"""Gradio web interface
"""

import gradio as gr
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import json
import time
from datetime import datetime

from researcher.researcher import AIResearcher
from researcher.llm import MODEL_PRESETS, get_model_preset


# ============================================================================
# Workflow Stage Definitions
# ============================================================================

WORKFLOW_STAGES = [
    {"name": "Task Parsing", "key": "task_parsing", "status": "implemented"},
    {"name": "Literature Review", "key": "literature_review", "status": "implemented"},
    {"name": "Hypothesis Construction", "key": "hypothesis_construction", "status": "implemented"},
    {"name": "Method Design", "key": "method_design", "status": "implemented"},
    {"name": "Experiment Execution", "key": "experiment_execution", "status": "todo"},
    {"name": "Report Generation", "key": "report_generation", "status": "implemented"},
    {"name": "Review", "key": "review", "status": "implemented"},
]


# ============================================================================
# Global State Management
# ============================================================================

class AppState:
    """Manages application state across UI interactions"""

    def __init__(self):
        self.researcher: Optional[AIResearcher] = None
        self.current_stage: str = "idle"
        self.workspace_dir: Optional[Path] = None
        self.execution_log: List[str] = []

    def reset(self):
        self.researcher = None
        self.current_stage = "idle"
        self.workspace_dir = None
        self.execution_log = []


app_state = AppState()


# ============================================================================
# UI Helper Functions
# ============================================================================

def generate_workflow_html(current_stage: Optional[str] = None, completed_stages: List[str] = None) -> str:
    """Generate HTML visualization of workflow progress"""
    if completed_stages is None:
        completed_stages = []

    css = """
    <style>
        .workflow-container {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            padding: 24px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 12px;
            color: white;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .workflow-title {
            font-size: 20px;
            font-weight: 600;
            margin-bottom: 24px;
            letter-spacing: 0.5px;
            text-align: center;
        }
        .stage-item {
            display: flex;
            align-items: center;
            padding: 14px 18px;
            margin: 10px 0;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            border-left: 4px solid transparent;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
        }
        .stage-item.completed {
            background: rgba(76, 175, 80, 0.25);
            border-left-color: #4CAF50;
        }
        .stage-item.current {
            background: rgba(33, 150, 243, 0.35);
            border-left-color: #2196F3;
            box-shadow: 0 2px 12px rgba(33, 150, 243, 0.4);
        }
        .stage-item.todo {
            background: rgba(255, 255, 255, 0.05);
            color: rgba(255, 255, 255, 0.45);
            border-left-color: rgba(255, 255, 255, 0.15);
        }
        .stage-number {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 32px;
            height: 32px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.2);
            font-weight: 600;
            font-size: 14px;
            margin-right: 14px;
            flex-shrink: 0;
        }
        .stage-item.completed .stage-number {
            background: #4CAF50;
        }
        .stage-item.current .stage-number {
            background: #2196F3;
        }
        .stage-item.todo .stage-number {
            background: rgba(255, 255, 255, 0.08);
        }
        .stage-name {
            flex: 1;
            font-size: 15px;
            font-weight: 500;
        }
        .stage-status {
            font-size: 12px;
            padding: 5px 12px;
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.15);
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .stage-item.completed .stage-status {
            background: rgba(76, 175, 80, 0.35);
        }
        .stage-item.current .stage-status {
            background: rgba(33, 150, 243, 0.35);
        }
        .stage-item.todo .stage-status {
            background: rgba(255, 255, 255, 0.08);
        }
    </style>
    """

    html_parts = [css, "<div class='workflow-container'>"]
    html_parts.append("<div class='workflow-title'>Research Workflow Pipeline</div>")

    for i, stage in enumerate(WORKFLOW_STAGES):
        is_current = current_stage == stage["key"]
        is_completed = stage["key"] in completed_stages
        is_todo = stage["status"] == "todo"

        classes = ["stage-item"]
        if is_completed:
            classes.append("completed")
            status_text = "Completed"
        elif is_current:
            classes.append("current")
            status_text = "Running"
        elif is_todo:
            classes.append("todo")
            status_text = "Pending"
        else:
            status_text = "Ready"

        html_parts.append(f"""
            <div class='{" ".join(classes)}'>
                <span class='stage-number'>{i + 1}</span>
                <span class='stage-name'>{stage['name']}</span>
                <span class='stage-status'>{status_text}</span>
            </div>
        """)

    html_parts.append("</div>")
    return "".join(html_parts)


def generate_model_info_html(model_preset: str) -> str:
    """Generate HTML for model configuration info"""
    try:
        model_config = get_model_preset(model_preset)

        css = """
        <style>
            .model-info {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 8px;
                padding: 18px;
            }
            .model-info-title {
                font-size: 13px;
                font-weight: 600;
                color: #495057;
                margin-bottom: 14px;
                text-transform: uppercase;
                letter-spacing: 0.8px;
            }
            .model-info-grid {
                display: grid;
                grid-template-columns: auto 1fr;
                gap: 10px 18px;
            }
            .model-info-label {
                font-size: 13px;
                color: #6c757d;
                font-weight: 500;
            }
            .model-info-value {
                font-size: 13px;
                color: #212529;
                font-family: 'SF Mono', 'Monaco', 'Menlo', monospace;
            }
        </style>
        """

        html = f"""
        {css}
        <div class='model-info'>
            <div class='model-info-title'>Model Configuration</div>
            <div class='model-info-grid'>
                <span class='model-info-label'>Provider:</span>
                <span class='model-info-value'>{model_config.provider}</span>
                <span class='model-info-label'>Model:</span>
                <span class='model-info-value'>{model_config.model_name}</span>
                <span class='model-info-label'>Temperature:</span>
                <span class='model-info-value'>{model_config.temperature}</span>
                <span class='model-info-label'>Max Tokens:</span>
                <span class='model-info-value'>{model_config.max_tokens}</span>
            </div>
        </div>
        """
        return html
    except Exception as e:
        return f"<div style='color: #dc3545; padding: 12px;'>Error loading model info: {str(e)}</div>"


# ============================================================================
# Core Research Functions
# ============================================================================

def initialize_researcher(project_name: str, model_preset: str, clear_workspace: bool) -> Tuple[str, str, str]:
    """Initialize AI Researcher instance"""
    try:
        app_state.reset()

        if not project_name.strip():
            project_name = f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        app_state.researcher = AIResearcher(
            project_name=project_name,
            model_preset=model_preset,
            clear_workspace=clear_workspace
        )

        app_state.workspace_dir = app_state.researcher.workspace_dir
        app_state.execution_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Initialized project: {project_name}")
        app_state.execution_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Workspace: {app_state.workspace_dir}")

        log_text = "\n".join(app_state.execution_log)
        workflow_html = generate_workflow_html()
        status_msg = f"Researcher initialized successfully for project: {project_name}"

        return status_msg, log_text, workflow_html

    except Exception as e:
        error_msg = f"Initialization failed: {str(e)}"
        app_state.execution_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: {error_msg}")
        return error_msg, "\n".join(app_state.execution_log), generate_workflow_html()


def run_research(input_text: str) -> Tuple:
    """Execute research workflow with progress tracking"""
    if not app_state.researcher:
        error_msg = "Please initialize researcher first"
        return (error_msg, "\n".join(app_state.execution_log), generate_workflow_html(),
                "", "", "", "", "", "")

    if not input_text.strip():
        error_msg = "Please provide research task input"
        return (error_msg, "\n".join(app_state.execution_log), generate_workflow_html(),
                "", "", "", "", "", "")

    try:
        app_state.execution_log.append(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting research workflow")
        app_state.execution_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Task: {input_text[:100]}...")

        completed_stages = []

        # Execute workflow
        final_state = app_state.researcher.run(input_text)

        app_state.execution_log.append(f"\n[{datetime.now().strftime('%H:%M:%S')}] Research workflow completed")
        app_state.current_stage = "completed"

        # Mark all implemented stages as completed
        for stage in WORKFLOW_STAGES:
            if stage["status"] == "implemented":
                completed_stages.append(stage["key"])

        # Load artifacts
        task_md = load_artifact("task.md")
        literature_md = load_artifact("literature.md")
        idea_md = load_artifact("idea.md")
        method_md = load_artifact("method.md")
        results_md = load_artifact("results.md")
        referee_md = load_artifact("referee.md")

        status_msg = "Research completed successfully"
        log_text = "\n".join(app_state.execution_log)
        workflow_html = generate_workflow_html("completed", completed_stages)

        return (status_msg, log_text, workflow_html, task_md, literature_md,
                idea_md, method_md, results_md, referee_md)

    except Exception as e:
        error_msg = f"Execution failed: {str(e)}"
        app_state.execution_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: {error_msg}")
        return (error_msg, "\n".join(app_state.execution_log), generate_workflow_html(),
                "", "", "", "", "", "")


def load_artifact(filename: str) -> str:
    """Load artifact from workspace"""
    if not app_state.workspace_dir:
        return ""

    artifact_path = app_state.workspace_dir / filename
    if artifact_path.exists():
        try:
            with open(artifact_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"Error loading {filename}: {str(e)}"
    return f"{filename} not yet generated"


def load_workspace_files() -> str:
    """List all files in workspace"""
    if not app_state.workspace_dir or not app_state.workspace_dir.exists():
        return "No workspace initialized"

    files = []
    for path in sorted(app_state.workspace_dir.rglob("*")):
        if path.is_file():
            rel_path = path.relative_to(app_state.workspace_dir)
            size = path.stat().st_size
            files.append(f"{rel_path} ({size} bytes)")

    return "\n".join(files) if files else "No files generated yet"


# ============================================================================
# Gradio Interface
# ============================================================================

def create_interface() -> gr.Blocks:
    """Create Gradio interface"""

    custom_css = """
    .gradio-container {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    .main-header {
        text-align: center;
        padding: 32px 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    """

    with gr.Blocks(title="AI Researcher", theme=gr.themes.Soft(), css=custom_css) as demo:

        gr.HTML("""
        <div class='main-header'>
            <h1 style='margin: 0; font-size: 36px; font-weight: 700; letter-spacing: -0.5px;'>AI Researcher</h1>
            <p style='margin: 12px 0 0 0; font-size: 16px; opacity: 0.9; font-weight: 400;'>
                Modular Scientific Research Automation System
            </p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Configuration")

                project_name = gr.Textbox(
                    label="Project Name",
                    placeholder="my_research_project",
                    value=f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )

                model_preset = gr.Dropdown(
                    label="Model Preset",
                    choices=list(MODEL_PRESETS.keys()),
                    value="openai-gpt4o",
                    interactive=True
                )

                model_info = gr.HTML(label="Model Info")

                clear_workspace = gr.Checkbox(
                    label="Clear existing workspace",
                    value=False
                )

                init_btn = gr.Button("Initialize Researcher", variant="primary", size="lg")

            with gr.Column(scale=2):
                gr.Markdown("### Research Task")

                input_text = gr.Textbox(
                    label="Task Description",
                    placeholder="Enter your research question or task description...",
                    lines=6
                )

                run_btn = gr.Button("Run Research Workflow", variant="primary", size="lg")

                status_box = gr.Textbox(label="Status", interactive=False, lines=1)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Workflow Progress")
                workflow_viz = gr.HTML(value=generate_workflow_html())

            with gr.Column():
                gr.Markdown("### Execution Log")
                log_box = gr.Textbox(
                    label="",
                    lines=15,
                    interactive=False,
                    show_copy_button=True
                )

        gr.Markdown("---")
        gr.Markdown("## Research Artifacts")

        with gr.Tabs():
            with gr.Tab("Task"):
                task_output = gr.Markdown()

            with gr.Tab("Literature Review"):
                literature_output = gr.Markdown()

            with gr.Tab("Research Ideas"):
                idea_output = gr.Markdown()

            with gr.Tab("Method Design"):
                method_output = gr.Markdown()

            with gr.Tab("Experiment Results"):
                results_output = gr.Markdown()

            with gr.Tab("Peer Review"):
                referee_output = gr.Markdown()

            with gr.Tab("Workspace Files"):
                with gr.Row():
                    refresh_files_btn = gr.Button("Refresh File List", size="sm")
                files_output = gr.Textbox(
                    label="Generated Files",
                    lines=15,
                    interactive=False,
                    show_copy_button=True
                )

        # Event handlers
        model_preset.change(
            fn=generate_model_info_html,
            inputs=[model_preset],
            outputs=[model_info]
        )

        init_btn.click(
            fn=initialize_researcher,
            inputs=[project_name, model_preset, clear_workspace],
            outputs=[status_box, log_box, workflow_viz]
        )

        run_btn.click(
            fn=run_research,
            inputs=[input_text],
            outputs=[
                status_box, log_box, workflow_viz,
                task_output, literature_output, idea_output,
                method_output, results_output, referee_output
            ]
        )

        refresh_files_btn.click(
            fn=load_workspace_files,
            outputs=[files_output]
        )

        # Initialize model info on load
        demo.load(
            fn=generate_model_info_html,
            inputs=[model_preset],
            outputs=[model_info]
        )

    return demo


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
