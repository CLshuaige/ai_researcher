from pathlib import Path
from typing import Optional
from datetime import datetime
import time

from researcher.config import get_workspace_dir, update_model_config
from researcher.state import ResearchState
from researcher.graph.researcher_graph import build_researcher_graph
from researcher.utils import (
    initialize_workspace,
    save_markdown,
    load_markdown,
    get_artifact_path,
    save_session_metadata,
    load_session_metadata,
)
from researcher.schemas import ResearchIdea, ExperimentalMethod, ExperimentResult, ReviewReport


class AIResearcher:
    """
    Main class for AI-powered research automation system

    Manages the complete research workflow from task definition to paper review.
    Uses LangGraph for workflow orchestration and AG2 for multi-agent collaboration.

    Args:
        project_name: Name of the research project
        workspace_dir: Optional custom workspace directory
        clear_workspace: Whether to clear existing workspace on initialization
        model_preset: Optional model preset name from registry
            (e.g., "openai-gpt4o", "qwen-30b-local"). If None, uses the
            default preset from configuration.
    """

    def __init__(
        self,
        project_name: str = "research_project",
        workspace_dir: Optional[Path] = None,
        clear_workspace: bool = False,
        model_preset: Optional[str] = None,
        enable_human_in_loop: bool = False
    ):
        self.project_name = project_name
        self.enable_human_in_loop = enable_human_in_loop

        # Update model configuration if preset provided
        if model_preset is not None:
            from researcher.llm import get_model_preset
            model_config = get_model_preset(model_preset)
            update_model_config(model_config)
            print(f"Using model preset: {model_preset} ({model_config.provider}/{model_config.model_name})")

        # Setup workspace
        if workspace_dir:
            self.workspace_dir = Path(workspace_dir)
            self.workspace_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.workspace_dir = get_workspace_dir(project_name)

        if clear_workspace:
            import shutil
            if self.workspace_dir.exists():
                shutil.rmtree(self.workspace_dir)
            self.workspace_dir.mkdir(parents=True, exist_ok=True)

        initialize_workspace(self.workspace_dir)

        # Build graph with checkpointer and optional interrupt
        self.graph = build_researcher_graph(enable_human_in_loop=enable_human_in_loop)
        self.current_state: Optional[ResearchState] = None
        self.session_id: Optional[str] = None

    def run(self, input_text: str, input_file: Optional[Path] = None, config: dict = None) -> ResearchState:
        """
        Execute the complete research workflow

        Args:
            input_text: Initial research prompt or task description
            input_file: Optional path to input.md file
            config: Optional configuration dictionary

        Returns:
            Final research state with all artifacts
        """
        start_time = time.time()

        if input_file and input_file.exists():
            with open(input_file, 'r', encoding='utf-8') as f:
                input_text = f.read()

        input_path = get_artifact_path(self.workspace_dir, "input")
        save_markdown(input_text, input_path)

        # Generate session_id from workspace directory name
        self.session_id = self.workspace_dir.name

        initial_state: ResearchState = {
            "input_text": input_text,
            "task": None,
            "config": config,
            "literature": None,
            "idea": None,
            "method": None,
            "results": None,
            "paper": None,
            "referee": None,
            "workspace_dir": self.workspace_dir,
            "project_name": self.project_name,
            "stage": "initialization",
            "error": None,
            "session_id": self.session_id,
            "human_feedback": None,
        }

        # Save session metadata
        session_data = {
            "session_id": self.session_id,
            "project_name": self.project_name,
            "created_at": datetime.now().isoformat(),
            "status": "running",
            "stage": "initialization",
        }
        save_session_metadata(self.workspace_dir, session_data)

        print(f"Starting research workflow for project: {self.project_name}")
        print(f"Workspace: {self.workspace_dir}")
        print(f"Session ID: {self.session_id}")

        # Execute workflow with thread_id for checkpointing
        config = {"configurable": {"thread_id": self.session_id}}
        final_state = self.graph.invoke(initial_state, config=config)

        # Store state
        self.current_state = final_state

        # Update session metadata
        session_data["status"] = "completed" if not final_state.get("error") else "failed"
        session_data["stage"] = final_state["stage"]
        session_data["completed_at"] = datetime.now().isoformat()
        save_session_metadata(self.workspace_dir, session_data)

        # Report completion
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print(f"\nWorkflow completed in {minutes} min {seconds} sec")
        print(f"Final stage: {final_state['stage']}")

        return final_state

    # Getters
    def get_workspace_path(self) -> Path:
        """Get the workspace directory path"""
        return self.workspace_dir

    def get_artifact(self, artifact_name: str) -> Optional[str]:
        """Retrieve specific artifact content as string"""
        artifact_path = get_artifact_path(self.workspace_dir, artifact_name)
        return load_markdown(artifact_path)

    def get_idea(self) -> Optional[ResearchIdea]:
        """Get the research idea object"""
        return self.current_state["idea"] if self.current_state else None

    def get_method(self) -> Optional[ExperimentalMethod]:
        """Get the experimental method object"""
        return self.current_state["method"] if self.current_state else None

    def get_results(self) -> Optional[ExperimentResult]:
        """Get the experiment results object"""
        return self.current_state["results"] if self.current_state else None

    def get_referee(self) -> Optional[ReviewReport]:
        """Get the review report object"""
        return self.current_state["referee"] if self.current_state else None

    # Display methods
    def show_idea(self) -> None:
        """Display the research idea"""
        idea = self.get_idea()
        if idea:
            print(idea.to_markdown())
        else:
            print("No idea generated yet")

    def show_method(self) -> None:
        """Display the experimental method"""
        method = self.get_method()
        if method:
            print(method.to_markdown())
        else:
            print("No method generated yet")

    def show_results(self) -> None:
        """Display the experiment results"""
        results = self.get_results()
        if results:
            print(results.to_markdown())
        else:
            print("No results generated yet")

    def show_referee(self) -> None:
        """Display the review report"""
        referee = self.get_referee()
        if referee:
            print(referee.to_markdown())
        else:
            print("No review generated yet")

    def show_workspace(self) -> None:
        """Display workspace structure"""
        print(f"\nWorkspace: {self.workspace_dir}")
        print("\nGenerated artifacts:")
        for artifact in ["task", "literature", "idea", "method", "results", "paper", "referee"]:
            path = get_artifact_path(self.workspace_dir, artifact)
            if path.exists():
                print(f"  ✓ {artifact}: {path}")
            else:
                print(f"  ✗ {artifact}: not generated")

    # Human-in-the-loop methods
    def get_current_state(self):
        """Get current workflow state (for human-in-the-loop)"""
        if not self.session_id:
            return None
        config = {"configurable": {"thread_id": self.session_id}}
        return self.graph.get_state(config)

    def update_human_feedback(self, feedback: dict):
        """Update human feedback and continue workflow

        Args:
            feedback: Dict containing human responses (e.g., {"answers": [...]})
        """
        if not self.session_id:
            raise ValueError("No active session. Call run() first.")

        config = {"configurable": {"thread_id": self.session_id}}

        # Update state with human feedback
        self.graph.update_state(config, {"human_feedback": feedback})

        print(f"Human feedback updated: {feedback}")

    def continue_workflow(self) -> ResearchState:
        """Continue workflow after human feedback"""
        if not self.session_id:
            raise ValueError("No active session. Call run() first.")

        config = {"configurable": {"thread_id": self.session_id}}

        print("Continuing workflow...")
        final_state = self.graph.invoke(None, config=config)

        # Store state
        self.current_state = final_state

        # Update session metadata
        session_data = load_session_metadata(self.workspace_dir) or {}
        session_data["status"] = "completed" if not final_state.get("error") else "failed"
        session_data["stage"] = final_state["stage"]
        session_data["updated_at"] = datetime.now().isoformat()
        save_session_metadata(self.workspace_dir, session_data)

        print(f"Workflow continued. Current stage: {final_state['stage']}")

        return final_state

    @staticmethod
    def resume(workspace_dir: Path, enable_human_in_loop: bool = False) -> 'AIResearcher':
        """Resume a previous research session

        Args:
            workspace_dir: Path to existing workspace
            enable_human_in_loop: Whether to enable human-in-the-loop

        Returns:
            AIResearcher instance with restored session
        """
        workspace_dir = Path(workspace_dir)
        if not workspace_dir.exists():
            raise ValueError(f"Workspace not found: {workspace_dir}")

        session_data = load_session_metadata(workspace_dir)
        if not session_data:
            raise ValueError(f"No session metadata found in {workspace_dir}")

        project_name = session_data.get("project_name", "resumed_project")

        researcher = AIResearcher(
            project_name=project_name,
            workspace_dir=workspace_dir,
            enable_human_in_loop=enable_human_in_loop
        )

        researcher.session_id = session_data.get("session_id", workspace_dir.name)

        print(f"Resumed session: {researcher.session_id}")
        print(f"Last stage: {session_data.get('stage', 'unknown')}")

        return researcher
